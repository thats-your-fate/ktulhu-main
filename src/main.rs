use std::{fs, sync::Arc, time::Duration};

use axum::{
    http::{header::AUTHORIZATION, header::CONTENT_TYPE, HeaderName, HeaderValue, Method},
    Router,
};
use dotenvy::dotenv;
use tokio::net::TcpListener;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use ktulhuMain::db::DBLayer;
use ktulhuMain::inference::intent_router::logits_argmax;
use ktulhuMain::manager::ModelManager;
use ktulhuMain::ws::{self, AppState, InferenceWorker};
use ktulhuMain::{
    auth, external_api,
    inference::InferenceService,
    internal_api,
    payment::{self, PaymentService},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // -----------------------------------
    // Load .env before anything else
    // -----------------------------------
    dotenv().ok();
    dotenvy::from_filename("config/payment.env").ok();
    dotenvy::from_filename("config/llamacpp.env").ok();

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

    println!("5️⃣ Sanity check (classifier quick pass)");
    let router = models.intent_router.clone();
    let sanity_handle = tokio::task::spawn_blocking(move || {
        router.classify("machine learning is cool").and_then(|out| {
            let (speech_idx, _) = logits_argmax(&out.speech_act)?;
            let (expect_idx, _) = logits_argmax(&out.expectation)?;
            Ok((speech_idx, expect_idx))
        })
    });
    match tokio::time::timeout(Duration::from_secs(10), sanity_handle).await {
        Ok(Ok(Ok((speech, expect)))) => {
            println!(
                "🧪 classifier check → speech_act={} expectation={}",
                speech, expect
            );
        }
        Ok(Ok(Err(err))) => {
            println!("⚠️  classifier sanity check failed: {err}");
        }
        Ok(Err(join_err)) => {
            println!("⚠️  classifier sanity check panicked: {join_err}");
        }
        Err(_) => println!("⚠️  classifier sanity check timed out (>10s)"),
    }

    // -----------------------------------
    // Unified inference service
    // -----------------------------------
    let infer = Arc::new(InferenceService::new(models.mistral_llama.clone()));

    // -----------------------------------
    // Optional payment service (Stripe)
    // -----------------------------------
    let payment_service = PaymentService::from_env();
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

    let device_header =
        HeaderName::from_lowercase(b"x-device-hash").expect("invalid device hash header");
    let cors_layer = CorsLayer::new()
        .allow_origin(AllowOrigin::list(allowed_origins.clone()))
        .allow_methods(AllowMethods::list([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ]))
        .allow_headers(AllowHeaders::list([
            CONTENT_TYPE,
            AUTHORIZATION,
            device_header.clone(),
        ]))
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
