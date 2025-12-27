use crate::{
    embeddings::handler::{embed_handler, embed_handler_3060},
    ws::AppState,
};
use axum::{
    routing::{get, post},
    Router,
};

pub mod handlers;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/external/api/generate", post(handlers::generate))
        .route("/external/api/embeddings", post(embed_handler))
        .route("/external/api/embeddings/3060", post(embed_handler_3060))
        .route("/external/api/profile", get(handlers::profile))
        .route("/external/api/usage", get(handlers::generation_usage))
        .route(
            "/external/api/credentials/generate",
            post(handlers::generate_api_credentials),
        )
        .route(
            "/external/api/credentials",
            post(handlers::store_api_credentials),
        )
        .route(
            "/external/api/credentials/validate",
            post(handlers::validate_api_credentials),
        )
}
