use axum::{
    body::Body,
    http::{header, HeaderValue, Request, StatusCode},
    middleware::Next,
    response::Response,
};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use once_cell::sync::OnceCell;
use serde::Deserialize;
use std::{fs, path::Path};
use tracing::{error, warn};

const AUTH_FILE: &str = "internal_admin_auth.json";
static INTERNAL_AUTH: OnceCell<Option<InternalAuthConfig>> = OnceCell::new();

#[derive(Debug, Deserialize)]
struct InternalAuthConfig {
    username: String,
    password: String,
}

pub async fn require_internal_auth(req: Request<Body>, next: Next) -> Result<Response, StatusCode> {
    let Some(config) = auth_config() else {
        error!("internal admin credentials are missing; create internal_admin_auth.json");
        return Ok(internal_error_response());
    };

    let Some(header) = req.headers().get(header::AUTHORIZATION) else {
        return Ok(unauthorized_response());
    };

    let Ok(header_str) = header.to_str() else {
        return Ok(unauthorized_response());
    };

    if !header_str.starts_with("Basic ") {
        return Ok(unauthorized_response());
    }

    let encoded = &header_str[6..];
    let Ok(decoded) = BASE64.decode(encoded) else {
        return Ok(unauthorized_response());
    };

    let Ok(decoded_str) = String::from_utf8(decoded) else {
        return Ok(unauthorized_response());
    };

    let mut parts = decoded_str.splitn(2, ':');
    let username = parts.next().unwrap_or("");
    let password = parts.next().unwrap_or("");

    if username != config.username || password != config.password {
        return Ok(unauthorized_response());
    }

    Ok(next.run(req).await)
}

fn auth_config() -> Option<&'static InternalAuthConfig> {
    INTERNAL_AUTH
        .get_or_init(|| load_auth_config().ok())
        .as_ref()
}

fn load_auth_config() -> Result<InternalAuthConfig, std::io::Error> {
    let path = Path::new(AUTH_FILE);
    if !path.exists() {
        warn!(
            path = AUTH_FILE,
            "internal admin auth file not found; internal routes disabled"
        );
        return Err(std::io::Error::from(std::io::ErrorKind::NotFound));
    }

    let raw = fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|err| {
        error!(?err, "failed to parse internal admin auth file");
        std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid auth file")
    })
}

fn unauthorized_response() -> Response {
    let mut res = Response::new(Body::from("unauthorized"));
    *res.status_mut() = StatusCode::UNAUTHORIZED;
    res.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Basic realm=\"Internal\""),
    );
    res
}

fn internal_error_response() -> Response {
    let mut res = Response::new(Body::from("internal auth not configured"));
    *res.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
    res
}
