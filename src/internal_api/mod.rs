use crate::ws::handler::AppState;
use axum::{
    routing::{delete, get, post},
    Router,
};

pub mod handlers;
pub mod types;

use handlers::{delete_thread, get_thread, internal_generate, list_chats_by_device};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/internal/generate", post(internal_generate))
        .route("/internal/chat-thread/{chat_id}", get(get_thread))
        .route("/internal/chat-thread/{chat_id}", delete(delete_thread))
        .route(
            "/internal/chats/by-device/{device_hash}",
            get(list_chats_by_device),
        )
}
