use crate::ws::AppState;
use axum::{
    routing::{get, post},
    Router,
};

pub mod handlers;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/external/api/generate", post(handlers::generate))
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
