use std::{fs, sync::Arc};

use axum::{
    http::{header::AUTHORIZATION, header::CONTENT_TYPE, HeaderValue, Method},
    Router,
};
use tokio::net::TcpListener;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use dotenvy::dotenv;

mod attachments;
mod auth;
mod classifier;
mod conversation;
mod db;
mod embeddings;
mod external_api;
mod inference;
mod internal_api;
mod manager;
mod model;
mod payment;
mod prompts;
mod reasoning;
mod ws;

use db::DBLayer;
use manager::ModelManager;
use ws::{inference_worker::warmup_job, AppState, InferenceWorker};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // -----------------------------------
    // Load .env before anything else
    // -----------------------------------
    dotenv().ok();
    dotenvy::from_filename("config/payment.env").ok();

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
    // Shared DB
    // -----------------------------------
    let db = Arc::new(DBLayer::new("chatdb")?);

    // -----------------------------------
    // Load ML models
    // -----------------------------------
    let models = Arc::new(ModelManager::new().await?);

    println!("5️⃣ Sanity check (what you asked for)");
    let test = models
        .roberta
        .embed("machine learning is cool")
        .await?;
    println!(
        "🧪 embedding check → dim={} norm={:.4}",
        test.len(),
        test.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // -----------------------------------
    // Unified inference service
    // -----------------------------------
    let infer = Arc::new(crate::inference::InferenceService::new(
        models.mistral.clone(),
        models.phi.clone(),
    ));

    // -----------------------------------
    // Optional payment service (Stripe)
    // -----------------------------------
    let payment_service = payment::PaymentService::from_env();
    if payment_service.is_some() {
        println!("💳 Stripe checkout enabled via /payment/create-checkout-session");
    } else {
        println!("⚠️  Stripe env vars missing — payment routes disabled");
    }

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
        payment: payment_service,
    };

    // -----------------------------------
    // Routers
    // -----------------------------------
    let allowed_origins = load_allowed_origins("config/allowed_origins.txt");
    let allowed_origin_log = allowed_origins
        .iter()
        .filter_map(|v| v.to_str().ok().map(|s| s.to_string()))
        .collect::<Vec<_>>();
    println!("🌐 CORS allowed origins: {:?}", allowed_origin_log);

    let cors_layer = CorsLayer::new()
        .allow_origin(AllowOrigin::list(allowed_origins.clone()))
        .allow_methods(AllowMethods::list([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ]))
        .allow_headers(AllowHeaders::list([CONTENT_TYPE, AUTHORIZATION]))
        .allow_credentials(true);

    let app = Router::new()
        .merge(ws::ws_router())
        .merge(auth::router())
        .merge(internal_api::router())
        .merge(external_api::router())
        .merge(payment::router())
        .layer(cors_layer)
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

fn load_allowed_origins(path: &str) -> Vec<HeaderValue> {
    let default = default_allowed_origin_strings()
        .into_iter()
        .filter_map(to_header_value)
        .collect::<Vec<_>>();

    match fs::read_to_string(path) {
        Ok(contents) => {
            let mut origins = Vec::new();
            for line in contents.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }
                if let Some(value) = to_header_value(trimmed.to_string()) {
                    origins.push(value);
                }
            }
            if origins.is_empty() {
                default
            } else {
                origins
            }
        }
        Err(_) => default,
    }
}

fn default_allowed_origin_strings() -> Vec<String> {
    vec![
        "https://ktulhu.com".to_string(),
        "https://dev.ktulhu.com".to_string(),
        "https://app.ktulhu.com".to_string(),
        "https://devfrontend.ktulhu.com".to_string(),
        "http://localhost:5173".to_string(),
    ]
}

fn to_header_value(origin: String) -> Option<HeaderValue> {
    HeaderValue::from_str(&origin).ok()
}
