use crate::ws::AppState;
use axum::{
    routing::{delete, get},
    Router,
};

pub mod handlers;
use handlers::{
    admin_latest_messages, admin_overview, admin_page, delete_message, delete_thread, get_thread,
    list_chats_by_device, list_chats_by_user, list_messages_by_device, list_messages_for_chat,
    set_message_liked, update_summary,
};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/internal/chat-thread/{chat_id}", get(get_thread))
        .route("/internal/chat-thread/{chat_id}", delete(delete_thread))
        .route(
            "/internal/chat-thread/{chat_id}/summary",
            axum::routing::put(update_summary),
        )
        // Alias to match FE
        .route("/chat-thread/{chat_id}", get(get_thread))
        .route("/chat-thread/{chat_id}", delete(delete_thread))
        .route(
            "/internal/chat-thread/{chat_id}/message/{message_id}",
            delete(delete_message),
        )
        .route(
            "/internal/chat-thread/{chat_id}/message/{message_id}/liked",
            axum::routing::put(set_message_liked),
        )
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
        .route("/internal/admin", get(admin_page))
        .route("/internal/admin/overview", get(admin_overview))
        .route("/internal/admin/last", get(admin_latest_messages))
}
