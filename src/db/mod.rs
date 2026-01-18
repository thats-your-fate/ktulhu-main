use anyhow::Result;
use rocksdb::{Direction, IteratorMode, Options, DB};
use serde_json;

use crate::{
    inference::byte_decoder::tidy_decoded_text,
    model::{chat::Chat, message::Message, user::User, user_device::UserDevice},
};

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
    str,
};

const DEVICE_CHAT_INDEX_FLAG: &str = "device_chat_index:built";

pub struct DBLayer {
    db: DB,
}

impl DBLayer {
    pub fn new(path: &str) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, path)?;
        Ok(Self { db })
    }

    // ============================================================
    // MESSAGE STORAGE (CHAT-ORDERED)
    // ============================================================
    fn msg_key(chat_id: &str, ts: i64, id: &str) -> String {
        format!("chat:{}:msg:{:020}:{id}", chat_id, ts)
        // 020 → zero-padded timestamp for correct sorting
    }

    fn user_device_key(user_id: &str, device_id: &str) -> String {
        format!("user_device:{user_id}:{device_id}")
    }

    fn device_lookup_key(device_hash: &str) -> String {
        format!("device_lookup:{device_hash}")
    }

    fn device_chat_prefix(device_hash: &str) -> String {
        format!("device_chat:{device_hash}:")
    }

    fn device_chat_key(device_hash: &str, chat_id: &str) -> String {
        format!("{}{}", Self::device_chat_prefix(device_hash), chat_id)
    }

    fn add_chat_to_device_index(&self, device_hash: &str, chat_id: &str) -> Result<()> {
        if device_hash.is_empty() {
            return Ok(());
        }
        let key = Self::device_chat_key(device_hash, chat_id);
        self.db.put(key, chat_id.as_bytes())?;
        self.db.put(DEVICE_CHAT_INDEX_FLAG, b"1")?;
        Ok(())
    }

    fn remove_chat_from_device_index(&self, device_hash: &str, chat_id: &str) -> Result<()> {
        if device_hash.is_empty() {
            return Ok(());
        }
        let key = Self::device_chat_key(device_hash, chat_id);
        self.db.delete(key)?;
        Ok(())
    }

    async fn ensure_device_chat_index(&self) -> Result<()> {
        if self.db.get(DEVICE_CHAT_INDEX_FLAG)?.is_some() {
            return Ok(());
        }
        self.rebuild_device_chat_index().await
    }

    async fn rebuild_device_chat_index(&self) -> Result<()> {
        let prefix = "device_chat:";
        let mut delete_keys = Vec::new();

        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_bytes(), Direction::Forward))
        {
            let (key, _) = item?;
            let k = str::from_utf8(&key)?;
            if !k.starts_with(prefix) {
                break;
            }
            delete_keys.push(key);
        }

        for key in delete_keys {
            self.db.delete(key)?;
        }

        // Load chats once so we can insert keys after metadata exists
        let chats = self.list_chats().await?;
        for chat in &chats {
            if let Some(device_hash) = chat.device_hash.as_deref() {
                let key = Self::device_chat_key(device_hash, &chat.id);
                self.db.put(key, chat.id.as_bytes())?;
            }
        }

        self.db.put(DEVICE_CHAT_INDEX_FLAG, b"1")?;
        Ok(())
    }

    pub async fn save_message(&self, msg: &Message) -> Result<()> {
        let key = Self::msg_key(&msg.chat_id, msg.ts, &msg.id);
        let stored = normalize_message(msg.clone());
        let val = serde_json::to_vec(&stored)?;
        self.db.put(key, val)?;
        Ok(())
    }

    pub async fn list_messages_for_chat(&self, chat_id: &str) -> Result<Vec<Message>> {
        let prefix = format!("chat:{}:msg:", chat_id);
        let mut results = Vec::new();

        let iter = self
            .db
            .iterator(IteratorMode::From(prefix.as_bytes(), Direction::Forward));

        for item in iter {
            let (key, val) = item?;
            let k = str::from_utf8(&key)?;

            if !k.starts_with(&prefix) {
                break;
            }

            let msg: Message = serde_json::from_slice(&val)?;
            results.push(normalize_message(msg));
        }

        Ok(results)
    }

    fn find_message_entry(
        &self,
        chat_id: &str,
        message_id: &str,
    ) -> Result<Option<(Vec<u8>, Message)>> {
        let prefix = format!("chat:{}:msg:", chat_id);
        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_bytes(), Direction::Forward))
        {
            let (key, val) = item?;
            let k = str::from_utf8(&key)?;
            if !k.starts_with(&prefix) {
                break;
            }

            let msg: Message = serde_json::from_slice(&val)?;
            if msg.id == message_id {
                return Ok(Some((key.to_vec(), normalize_message(msg))));
            }
        }
        Ok(None)
    }

    pub async fn delete_message(&self, chat_id: &str, message_id: &str) -> Result<bool> {
        if let Some((key, _)) = self.find_message_entry(chat_id, message_id)? {
            self.db.delete(key)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub async fn set_message_liked(
        &self,
        chat_id: &str,
        message_id: &str,
        liked: bool,
    ) -> Result<bool> {
        if let Some((key, mut msg)) = self.find_message_entry(chat_id, message_id)? {
            msg.liked = liked;
            self.db.put(key, serde_json::to_vec(&msg)?)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Collect the latest raw messages across all chats, ordered by timestamp desc.
    pub async fn list_recent_messages(&self, limit: usize) -> Result<Vec<Message>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        #[derive(Copy, Clone, Eq, PartialEq)]
        struct HeapKey {
            ts: i64,
            seq: usize,
        }

        impl Ord for HeapKey {
            fn cmp(&self, other: &Self) -> Ordering {
                self.ts.cmp(&other.ts).then(self.seq.cmp(&other.seq))
            }
        }

        impl PartialOrd for HeapKey {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        struct HeapEntry {
            key: HeapKey,
            message: Message,
        }

        impl PartialEq for HeapEntry {
            fn eq(&self, other: &Self) -> bool {
                self.key == other.key
            }
        }

        impl Eq for HeapEntry {}

        impl Ord for HeapEntry {
            fn cmp(&self, other: &Self) -> Ordering {
                // Reverse so BinaryHeap pops the oldest (smallest ts) first.
                other.key.cmp(&self.key)
            }
        }

        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        let mut seq = 0usize;

        for item in self.db.iterator(IteratorMode::Start) {
            let (key, val) = item?;
            let key_str = str::from_utf8(&key)?;
            if !key_str.starts_with("chat:") || !key_str.contains(":msg:") {
                continue;
            }

            let msg: Message = serde_json::from_slice(&val)?;
            let msg = normalize_message(msg);
            seq = seq.wrapping_add(1);
            let entry = HeapEntry {
                key: HeapKey { ts: msg.ts, seq },
                message: msg,
            };

            if heap.len() < limit {
                heap.push(entry);
                continue;
            }

            if heap
                .peek()
                .map(|oldest| entry.key > oldest.key)
                .unwrap_or(false)
            {
                heap.pop();
                heap.push(entry);
            }
        }

        let mut messages: Vec<_> = heap.into_iter().map(|entry| entry.message).collect();
        messages.sort_by(|a, b| b.ts.cmp(&a.ts).then_with(|| a.id.cmp(&b.id)));
        Ok(messages)
    }

    // ============================================================
    // CHAT STORAGE
    // ============================================================
    pub async fn save_chat(&self, chat: &Chat) -> Result<()> {
        let key = format!("chat:meta:{}", chat.id);
        let previous_chat: Option<Chat> = self
            .db
            .get(&key)?
            .map(|val| serde_json::from_slice::<Chat>(&val))
            .transpose()?;

        match (
            previous_chat
                .as_ref()
                .and_then(|c| c.device_hash.as_deref()),
            chat.device_hash.as_deref(),
        ) {
            (Some(old_hash), Some(new_hash)) if old_hash != new_hash => {
                self.remove_chat_from_device_index(old_hash, &chat.id)?;
                self.add_chat_to_device_index(new_hash, &chat.id)?;
            }
            (Some(old_hash), None) => {
                self.remove_chat_from_device_index(old_hash, &chat.id)?;
            }
            (None, Some(new_hash)) => {
                self.add_chat_to_device_index(new_hash, &chat.id)?;
            }
            _ => {}
        }

        let val = serde_json::to_vec(chat)?;
        self.db.put(key, val)?;
        Ok(())
    }

    pub async fn load_chat(&self, id: &str) -> Result<Option<Chat>> {
        let key = format!("chat:meta:{id}");
        Ok(self
            .db
            .get(key)?
            .map(|v| serde_json::from_slice(&v).unwrap()))
    }

    pub async fn list_chats(&self) -> Result<Vec<Chat>> {
        let prefix = "chat:meta:";
        let mut results = Vec::new();

        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_bytes(), Direction::Forward))
        {
            let (key, val) = item?;
            let k = str::from_utf8(&key)?;

            if !k.starts_with(prefix) {
                break;
            }

            let chat: Chat = serde_json::from_slice(&val)?;
            results.push(chat);
        }

        Ok(results)
    }

    /// List all chats belonging to all devices of a user.
    pub async fn list_chats_for_user(&self, user_id: &str) -> Result<Vec<Chat>> {
        // 1. Load all devices for user
        let devices = self.list_devices_for_user(user_id).await?;

        let mut all_chats = Vec::new();
        let mut seen_ids = HashSet::new();

        // 2. For each device, load its chats
        for device in devices {
            let chats = self.list_chats_for_device(&device.device_hash).await?;
            for chat in chats {
                if seen_ids.insert(chat.id.clone()) {
                    all_chats.push(chat);
                }
            }
        }

        Ok(all_chats)
    }

    pub async fn list_chats_for_device(&self, device_hash: &str) -> Result<Vec<Chat>> {
        if device_hash.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_device_chat_index().await?;

        let prefix = Self::device_chat_prefix(device_hash);
        let mut chat_ids = Vec::new();

        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_bytes(), Direction::Forward))
        {
            let (key, _) = item?;
            let k = str::from_utf8(&key)?;
            if !k.starts_with(&prefix) {
                break;
            }
            chat_ids.push(k[prefix.len()..].to_string());
        }

        let mut chats = Vec::with_capacity(chat_ids.len());
        for chat_id in chat_ids {
            match self.load_chat(&chat_id).await? {
                Some(chat) => chats.push(chat),
                None => {
                    self.remove_chat_from_device_index(device_hash, &chat_id)?;
                }
            }
        }

        Ok(chats)
    }

    /// Delete all messages (and chat metadata) for a chat id.
    pub async fn delete_thread(&self, chat_id: &str) -> Result<()> {
        let existing_chat = self.load_chat(chat_id).await?;
        let prefix = format!("chat:{}:msg:", chat_id);

        // Collect keys first to avoid mutating while iterating.
        let mut keys = Vec::new();
        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_bytes(), Direction::Forward))
        {
            let (key, _) = item?;
            let k_str = str::from_utf8(&key)?;
            if !k_str.starts_with(&prefix) {
                break;
            }
            keys.push(key);
        }

        for key in keys {
            self.db.delete(key)?;
        }

        // Remove chat metadata if present.
        let meta_key = format!("chat:meta:{chat_id}");
        let _ = self.db.delete(meta_key);

        if let Some(chat) = existing_chat {
            if let Some(device_hash) = chat.device_hash.as_deref() {
                self.remove_chat_from_device_index(device_hash, chat_id)?;
            }
        }

        Ok(())
    }

    pub async fn remove_messages_by_role(&self, chat_id: &str, role: &str) -> Result<usize> {
        let prefix = format!("chat:{}:msg:", chat_id);
        let mut keys = Vec::new();

        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_bytes(), Direction::Forward))
        {
            let (key, val) = item?;
            let k = str::from_utf8(&key)?;
            if !k.starts_with(&prefix) {
                break;
            }

            let msg: Message = serde_json::from_slice(&val)?;
            if msg.role == role {
                keys.push(key);
            }
        }

        for key in &keys {
            self.db.delete(key)?;
        }

        Ok(keys.len())
    }

    // ============================================================
    // USER STORAGE
    // ============================================================
    pub async fn save_user(&self, user: &User) -> Result<()> {
        let key = format!("user:{}", user.id);
        let val = serde_json::to_vec(user)?;
        self.db.put(key, val)?;
        Ok(())
    }

    pub async fn load_user(&self, id: &str) -> Result<Option<User>> {
        let key = format!("user:{id}");
        Ok(self
            .db
            .get(key)?
            .map(|v| serde_json::from_slice(&v).unwrap()))
    }

    pub async fn list_users(&self) -> Result<Vec<User>> {
        let prefix = "user:";
        let mut results = Vec::new();

        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_bytes(), Direction::Forward))
        {
            let (key, val) = item?;
            let k = std::str::from_utf8(&key)?;

            if !k.starts_with(prefix) {
                break;
            }

            let user: User = serde_json::from_slice(&val)?;
            results.push(user);
        }

        Ok(results)
    }

    pub async fn list_devices_for_user(&self, user_id: &str) -> Result<Vec<UserDevice>> {
        let prefix = format!("user_device:{user_id}:");
        let mut out = Vec::new();

        for item in self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        )) {
            let (key, val) = item?;
            let k = std::str::from_utf8(&key)?;
            if !k.starts_with(&prefix) {
                break;
            }
            out.push(serde_json::from_slice(&val)?);
        }
        Ok(out)
    }

    pub async fn delete_user(&self, user_id: &str) -> Result<()> {
        let user_key = format!("user:{user_id}");
        self.db.delete(user_key)?;

        if let Ok(devices) = self.list_devices_for_user(user_id).await {
            for device in devices {
                let key = Self::user_device_key(&device.user_id, &device.id);
                self.db.delete(key)?;
                if !device.device_hash.is_empty() {
                    let lookup_key = Self::device_lookup_key(&device.device_hash);
                    self.db.delete(lookup_key)?;
                }
            }
        }

        Ok(())
    }

    pub async fn list_all_devices(&self) -> Result<Vec<UserDevice>> {
        let prefix = "user_device:";
        let mut out = Vec::new();

        for item in self
            .db
            .iterator(IteratorMode::From(prefix.as_bytes(), Direction::Forward))
        {
            let (key, val) = item?;
            let k = std::str::from_utf8(&key)?;
            if !k.starts_with(prefix) {
                break;
            }
            out.push(serde_json::from_slice(&val)?);
        }

        Ok(out)
    }

    pub async fn add_device_for_user(&self, user_id: &str, device_hash: &str) -> Result<()> {
        let dev = UserDevice {
            id: uuid::Uuid::new_v4().to_string(),
            user_id: user_id.to_string(),
            device_hash: device_hash.to_string(),
            created_ts: chrono::Utc::now().timestamp(),
            last_seen_ts: chrono::Utc::now().timestamp(),
            meta: None,
        };

        let key = Self::user_device_key(user_id, &dev.id);
        let val = serde_json::to_vec(&dev)?;
        self.db.put(key, val)?;

        // fast lookup: device → user
        let lookup_key = Self::device_lookup_key(device_hash);
        self.db.put(lookup_key, user_id)?;

        Ok(())
    }
}

fn normalize_message(mut msg: Message) -> Message {
    normalize_option_text(&mut msg.text);
    for attachment in msg.attachments.iter_mut() {
        normalize_option_text(&mut attachment.description);
        normalize_option_text(&mut attachment.ocr_text);
    }
    msg
}

fn normalize_option_text(target: &mut Option<String>) {
    if let Some(text) = target.take() {
        let normalized = tidy_decoded_text(&text);
        target.replace(normalized);
    }
}
