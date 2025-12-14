use crate::ws::AppState;
use axum::{
    routing::{delete, get, post},
    Router,
};

pub mod handlers;
pub mod storage;

use handlers::{
    delete_thread, get_thread, list_chats_by_device, list_chats_by_user, list_messages_by_device,
    list_messages_for_chat,
};
use storage::{get_file, upload_file};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/internal/chat-thread/{chat_id}", get(get_thread))
        .route("/internal/chat-thread/{chat_id}", delete(delete_thread))
        // Alias to match FE
        .route("/chat-thread/{chat_id}", get(get_thread))
        .route("/chat-thread/{chat_id}", delete(delete_thread))
        .route(
            "/internal/chats/by-device/{device_hash}",
            get(list_chats_by_device),
        )
        .route(
            "/internal/messages/by-device/{device_hash}",
            get(list_messages_by_device),
        )
        // *** NEW: aggregate user chats across all devices ***
        .route("/internal/chats/by-user/{user_id}", get(list_chats_by_user))
        // Former external API endpoints
        .route("/api/chats/{chat_id}/messages", get(list_messages_for_chat))
        .route("/api/storage/upload", post(upload_file))
        .route("/api/storage/files/{filename}", get(get_file))
}
