pub mod apple;
pub mod email_auth;
pub mod google;
pub mod google_keys;
pub mod jwt;
pub mod types;
pub mod utils;
use crate::ws::AppState;
use axum::{routing::post, Router};

use crate::auth::email_auth::{email_login_handler, email_register_handler};

/// Full auth router: Google + Apple + Facebook
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/auth/google", post(google::google_login_handler))
        .route("/api/auth/apple", post(apple::apple_login_handler))
        .route("/api/auth/register", post(email_register_handler))
        .route("/api/auth/login", post(email_login_handler))
}
