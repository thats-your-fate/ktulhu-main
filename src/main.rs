use std::sync::Arc;

use axum::Router;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod conversation;
mod db;
mod inference;
mod internal_api;
mod model;
mod ws;

use db::DBLayer;
use inference::mistral::InferenceService;
use ws::handler::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // -----------------------------
    // Logging
    // -----------------------------
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .init();

    println!("🚀 Starting Mistral inference server...");

    // -----------------------------
    // Shared state / Dependencies
    // -----------------------------
    let db = Arc::new(DBLayer::new("chatdb")?);
    let infer = Arc::new(InferenceService::new().await?);

    let state = AppState { db, infer };

    // -----------------------------
    // JWT secret — removed
    // -----------------------------
    /*
    let jwt_secret = std::env::var("JWT_SECRET")
        .unwrap_or_else(|_| "supersecret123".into());
    */

    // -----------------------------
    // Routers
    // -----------------------------
    let app = Router::new()
        // WebSocket inference
        .merge(ws::ws_router())
        // 🔥 JWT API removed
        // .merge(api::api_router(jwt_secret.clone()))
        // Internal API (no JWT, server-only)
        .merge(internal_api::router())
        // CORS for frontend
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_headers(Any)
                .allow_methods(Any),
        )
        // Attach shared state
        .with_state(state);

    let addr = "0.0.0.0:3000";

    println!("🌐 HTTP listening on http://{addr}");
    println!("🔌 WebSocket at ws://{addr}/ws");
    println!("🛠 Internal API at http://{addr}/internal/generate");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app.into_make_service()).await?;

    Ok(())
}
