use anyhow::{anyhow, Result};
use axum::{http::StatusCode, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use axum_extra::typed_header::TypedHeader;
use headers::{authorization::Bearer, Authorization};

use crate::{
    auth::jwt::decode_jwt,
    model::user::{User, UserRole},
    ws::AppState,
};

/// Payment helper for wiring Stripe Checkout without pulling in the entire Go example.
///
/// How to use from the frontend:
/// 1. Configure the env vars `STRIPE_PUBLISHABLE_KEY`, `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_SUCCESS_URL`, and `STRIPE_CANCEL_URL`.
/// 2. POST to `/payment/create-checkout-session` (requires no body) and read the JSON response.
/// 3. Redirect the browser to `checkout_url` to hand over to Stripe. After the customer finishes, Stripe sends them
///    back to `STRIPE_SUCCESS_URL` (or the cancel URL if they exit).
/// 4. Listen to Stripe webhooks / dashboard to confirm payment and upgrade the user role through the admin tools.
#[derive(Clone)]
pub struct PaymentService {
    client: reqwest::Client,
    secret_key: String,
    publishable_key: String,
    price_id: String,
    checkout_mode: String,
    success_url: String,
    cancel_url: String,
}

impl PaymentService {
    pub fn from_env() -> Option<Self> {
        let secret_key = dotenvy::var("STRIPE_SECRET_KEY").ok()?;
        let publishable_key = dotenvy::var("STRIPE_PUBLISHABLE_KEY").ok()?;
        let price_id = dotenvy::var("STRIPE_PRICE_ID").ok()?;
        let checkout_mode =
            dotenvy::var("STRIPE_CHECKOUT_MODE").unwrap_or_else(|_| "subscription".to_string());
        let success_url = dotenvy::var("STRIPE_SUCCESS_URL")
            .unwrap_or_else(|_| "http://localhost:3000/payment/success".to_string());
        let cancel_url = dotenvy::var("STRIPE_CANCEL_URL")
            .unwrap_or_else(|_| "http://localhost:3000/payment/cancel".to_string());

        Some(Self {
            client: reqwest::Client::new(),
            secret_key,
            publishable_key,
            price_id,
            checkout_mode,
            success_url,
            cancel_url,
        })
    }

    async fn create_checkout_session(&self, user_id: &str) -> Result<StripeCheckoutSession> {
        let mut form = Vec::new();
        form.push(("mode".to_string(), self.checkout_mode.clone()));
        form.push((
            "success_url".to_string(),
            self.success_url_with_session_placeholder(),
        ));
        form.push(("cancel_url".to_string(), self.cancel_url.clone()));
        if self.checkout_mode == "payment" {
            form.push(("customer_creation".to_string(), "always".to_string()));
        }
        form.push(("automatic_tax[enabled]".to_string(), "true".to_string()));
        form.push(("line_items[0][price]".to_string(), self.price_id.clone()));
        form.push(("line_items[0][quantity]".to_string(), "1".to_string()));
        form.push(("metadata[user_id]".to_string(), user_id.to_string()));

        let response = self
            .client
            .post("https://api.stripe.com/v1/checkout/sessions")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", self.secret_key),
            )
            .form(&form)
            .send()
            .await?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("stripe_error: {}", text));
        }

        let session: StripeCheckoutSession = response.json().await?;
        Ok(session)
    }

    async fn retrieve_checkout_session(&self, session_id: &str) -> Result<StripeSessionDetails> {
        let url = format!("https://api.stripe.com/v1/checkout/sessions/{}", session_id);
        let response = self
            .client
            .get(url)
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", self.secret_key),
            )
            .send()
            .await?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("stripe_error: {}", text));
        }

        Ok(response.json().await?)
    }

    fn success_url_with_session_placeholder(&self) -> String {
        if self.success_url.contains("{CHECKOUT_SESSION_ID}") {
            return self.success_url.clone();
        }

        if self.success_url.contains('?') {
            format!(
                "{}&session_id={{CHECKOUT_SESSION_ID}}",
                self.success_url
            )
        } else {
            format!(
                "{}?session_id={{CHECKOUT_SESSION_ID}}",
                self.success_url
            )
        }
    }
}

#[derive(Deserialize)]
struct StripeCheckoutSession {
    id: String,
    url: Option<String>,
}

#[derive(Deserialize)]
struct StripeSessionDetails {
    status: Option<String>,
    payment_status: Option<String>,
    #[serde(default)]
    metadata: HashMap<String, String>,
    subscription: Option<String>,
    customer: Option<String>,
}

#[derive(Serialize)]
pub struct CheckoutSessionResponse {
    pub session_id: String,
    pub checkout_url: String,
}

#[derive(Serialize)]
pub struct PaymentConfigResponse {
    pub publishable_key: String,
}

#[derive(Deserialize)]
struct ActivateRequest {
    session_id: String,
}

#[derive(Serialize)]
struct ActivateResponse {
    user_id: String,
    role: UserRole,
    updated: bool,
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route(
            "/payment/create-checkout-session",
            post(create_checkout_session),
        )
        .route("/payment/config", axum::routing::get(payment_config))
        .route("/payment/activate", post(activate_subscription))
}

async fn create_checkout_session(
    axum::extract::State(state): axum::extract::State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
) -> Result<Json<CheckoutSessionResponse>, (StatusCode, String)> {
    let user = authenticate_user(&state, auth.token()).await?;
    let service = state.payment.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "payments_not_configured".to_string(),
    ))?;

    let session = service
        .create_checkout_session(&user.id)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    let checkout_url = session
        .url
        .clone()
        .ok_or((StatusCode::BAD_GATEWAY, "missing_checkout_url".to_string()))?;

    Ok(Json(CheckoutSessionResponse {
        session_id: session.id,
        checkout_url,
    }))
}

async fn payment_config(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Result<Json<PaymentConfigResponse>, (StatusCode, String)> {
    let service = state.payment.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "payments_not_configured".to_string(),
    ))?;

    Ok(Json(PaymentConfigResponse {
        publishable_key: service.publishable_key.clone(),
    }))
}

async fn activate_subscription(
    axum::extract::State(state): axum::extract::State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    Json(payload): Json<ActivateRequest>,
) -> Result<Json<ActivateResponse>, (StatusCode, String)> {
    let auth_user = authenticate_user(&state, auth.token()).await?;
    if payload.session_id.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "session_id_required".to_string()));
    }
    let service = state.payment.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "payments_not_configured".to_string(),
    ))?;

    let session = service
        .retrieve_checkout_session(&payload.session_id)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    let owner_user_id = session
        .metadata
        .get("user_id")
        .cloned()
        .ok_or((StatusCode::BAD_REQUEST, "session_missing_user".to_string()))?;

    if owner_user_id != auth_user.id {
        return Err((StatusCode::FORBIDDEN, "session_owner_mismatch".to_string()));
    }

    if !session_is_paid(&session) {
        return Err((StatusCode::BAD_REQUEST, "session_not_paid".to_string()));
    }

    let mut user = state
        .db
        .load_user(&owner_user_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "user_not_found".to_string()))?;

    let mut updated = false;

    if user.role != UserRole::Paid {
        user.role = UserRole::Paid;
        updated = true;
    }

    if let Some(customer_id) = session.customer.clone() {
        if user.stripe_customer_id.as_deref() != Some(customer_id.as_str()) {
            user.stripe_customer_id = Some(customer_id);
            updated = true;
        }
    }

    if let Some(subscription_id) = session.subscription.clone() {
        if user
            .stripe_subscription_id
            .as_deref()
            != Some(subscription_id.as_str())
        {
            user.stripe_subscription_id = Some(subscription_id);
            updated = true;
        }
    }

    if updated {
        state
            .db
            .save_user(&user)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }

    Ok(Json(ActivateResponse {
        user_id: user.id,
        role: user.role,
        updated,
    }))
}

fn session_is_paid(session: &StripeSessionDetails) -> bool {
    matches!(
        session.status.as_deref(),
        Some("complete" | "complete_async")
    ) || matches!(session.payment_status.as_deref(), Some("paid"))
}

async fn authenticate_user(state: &AppState, token: &str) -> Result<User, (StatusCode, String)> {
    let user_id = decode_jwt(token, &state.jwt_secret)
        .map_err(|_| (StatusCode::UNAUTHORIZED, "invalid_token".into()))?;

    let user = state
        .db
        .load_user(&user_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::UNAUTHORIZED, "user_not_found".to_string()))?;

    Ok(user)
}
