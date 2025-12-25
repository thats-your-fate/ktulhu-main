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
}
