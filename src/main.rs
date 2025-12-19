use std::sync::Arc;

use axum::Router;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use dotenvy::dotenv;

mod auth;
mod classifier;
mod conversation;
mod db;
mod inference;
mod internal_api;
mod manager;
mod model;
mod prompts;
mod reasoning;
mod storage;
mod ws;

use db::DBLayer;
use manager::ModelManager;
use storage::StorageService;
use ws::{inference_worker::warmup_job, AppState, InferenceWorker};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // -----------------------------------
    // Load .env before anything else
    // -----------------------------------
    dotenv().ok();

    // -----------------------------------
    // Logging
    // -----------------------------------
    let log_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tokio_tungstenite=warn,tungstenite=warn"));

    tracing_subscriber::registry()
        .with(log_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    println!("\n🌟 Starting multi-model inference server…\n");

    // -----------------------------------
    // Environment variables
    // -----------------------------------
    let jwt_secret = dotenvy::var("JWT_SECRET").expect("❌ JWT_SECRET must be set in .env");

    let google_client_id = dotenvy::var("GOOGLE_CLIENT_ID").unwrap_or_else(|_| {
        println!("⚠️  GOOGLE_CLIENT_ID not set — Google Login disabled");
        String::new()
    });

    let apple_client_id = dotenvy::var("APPLE_CLIENT_ID").unwrap_or_else(|_| {
        println!("⚠️  APPLE_CLIENT_ID not set — Apple Login disabled");
        String::new()
    });

    // -----------------------------------
    // Local storage for uploads
    // -----------------------------------
    let storage = Arc::new(StorageService::new("storage").await?);

    // -----------------------------------
    // Shared DB
    // -----------------------------------
    let db = Arc::new(DBLayer::new("chatdb")?);

    // -----------------------------------
    // Load ML models
    // -----------------------------------
    let models = Arc::new(ModelManager::new().await?);

    // -----------------------------------
    // Unified inference service
    // -----------------------------------
    let infer = Arc::new(crate::inference::InferenceService::new(
        models.mistral.clone(),
        models.phi.clone(),
        models.roberta.clone(),
    ));

    // -----------------------------------
    // WebSocket inference worker
    // -----------------------------------
    let worker = InferenceWorker::new(16);

    // -----------------------------------
    // 🔥 Warm up inference worker
    // -----------------------------------
    {
        let infer = infer.clone();
        let db = db.clone();
        let worker_clone = worker.clone();

        tokio::spawn(async move {
            println!("🔥 Warming up inference worker…");

            if let Err(e) = worker_clone.enqueue(warmup_job(infer, db)).await {
                eprintln!("warmup enqueue failed: {e}");
            }
        });
    }

    // -----------------------------------
    // Global AppState
    // -----------------------------------
    let state = AppState {
        db,
        models,
        infer,
        worker,
        jwt_secret,
        google_client_id,
        apple_client_id,
        storage,
    };

    // -----------------------------------
    // Routers
    // -----------------------------------
    let app = Router::new()
        .merge(ws::ws_router())
        .merge(auth::router())
        .merge(internal_api::router())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_headers(Any)
                .allow_methods(Any),
        )
        .with_state(state);

    // -----------------------------------
    // Startup info
    // -----------------------------------
    let addr = "0.0.0.0:3000";

    println!("🌍 HTTP server  → http://{addr}");
    println!("🔌 WebSocket    → ws://{addr}/ws");
    println!("🔐 Auth API     → http://{addr}/api/auth/google");
    println!("🧠 Internal API → http://{addr}/internal\n");

    // -----------------------------------
    // Bind + serve
    // -----------------------------------
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app.into_make_service()).await?;

    Ok(())
}
