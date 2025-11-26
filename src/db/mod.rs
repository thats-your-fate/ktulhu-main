use anyhow::Result;
use rocksdb::{Direction, IteratorMode, Options, DB};
use serde_json;

use crate::model::{chat::Chat, message::Message, user::User};
use std::str;

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

    pub async fn save_message(&self, msg: &Message) -> Result<()> {
        let key = Self::msg_key(&msg.chat_id, msg.ts, &msg.id);
        let val = serde_json::to_vec(msg)?;
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
            results.push(msg);
        }

        Ok(results)
    }

    /// Get last N messages (for context pruning)
    pub async fn list_last_messages(&self, chat_id: &str, n: usize) -> Result<Vec<Message>> {
        let prefix = format!("chat:{}:msg:", chat_id);

        // Reverse iterator
        let iter = self.db.iterator(IteratorMode::End);
        let mut collected = Vec::new();

        for item in iter {
            let (key, val) = item?;
            let k = str::from_utf8(&key)?;

            if k.starts_with(&prefix) {
                let msg: Message = serde_json::from_slice(&val)?;
                collected.push(msg);

                if collected.len() >= n {
                    break;
                }
            }
        }

        collected.reverse();
        Ok(collected)
    }

    // ============================================================
    // CHAT STORAGE
    // ============================================================
    pub async fn save_chat(&self, chat: &Chat) -> Result<()> {
        let key = format!("chat:meta:{}", chat.id);
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

    pub async fn list_chats_for_user(&self, user_id: &str) -> Result<Vec<Chat>> {
        Ok(self
            .list_chats()
            .await?
            .into_iter()
            .filter(|c| c.user_id.as_deref() == Some(user_id))
            .collect())
    }

    pub async fn list_chats_for_device(&self, device_hash: &str) -> Result<Vec<Chat>> {
        Ok(self
            .list_chats()
            .await?
            .into_iter()
            .filter(|c| c.device_hash.as_deref() == Some(device_hash))
            .collect())
    }

    /// Delete all messages (and chat metadata) for a chat id.
    pub async fn delete_thread(&self, chat_id: &str) -> Result<()> {
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

        Ok(())
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
}
