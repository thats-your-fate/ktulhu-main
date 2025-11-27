use axum::{
    routing::{get, post},
    Extension, Router,
};

use crate::ws::AppState;

pub mod handlers;
pub mod types;

use handlers::{generate_handler, list_messages_for_chat};

/// Public API router (JWT protected)
pub fn api_router(secret: String) -> Router<AppState> {
    Router::new()
        // GET /api/chats/:chat_id/messages
        .route("/api/chats/{chat_id}/messages", get(list_messages_for_chat))
}
