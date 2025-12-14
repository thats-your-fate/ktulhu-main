use axum::{
    body::to_bytes,
    extract::{Multipart, Path, Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    Json,
};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::Deserialize;

use crate::{storage::StoredFile, ws::AppState};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum UploadRequest {
    Single(UploadRequestSingle),
    Multi(UploadRequestMulti),
}

#[derive(Debug, Deserialize)]
pub struct UploadRequestSingle {
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    pub data_base64: String,
}

#[derive(Debug, Deserialize)]
pub struct UploadRequestMulti {
    #[serde(default)]
    pub files: Vec<UploadRequestSingle>,
}

#[derive(Debug, serde::Serialize)]
pub struct UploadResponse {
    pub filename: String,
    pub mime_type: Option<String>,
    pub url: String,
}

pub async fn upload_file(
    State(state): State<AppState>,
    req: Request,
) -> Result<Json<UploadResponse>, (StatusCode, String)> {
    let (stored, mime_type) = if is_multipart(req.headers()) {
        let mut multipart =
            <Multipart as axum::extract::FromRequest<AppState>>::from_request(req, &state)
                .await
                .map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!(
                            "Invalid multipart payload (check Content-Type boundary or use JSON upload): {e}"
                        ),
                    )
                })?;

        let mut stored: Option<StoredFile> = None;
        let mut mime_type: Option<String> = None;

        while let Some(field) = multipart.next_field().await.map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("Invalid multipart payload: {e}"),
            )
        })? {
            if stored.is_some() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "Only one file per request is supported on this endpoint".to_string(),
                ));
            }

            let file_name = field.file_name().map(|s| s.to_string());
            mime_type = field.content_type().map(|m| m.to_string());

            let bytes = field.bytes().await.map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to read field: {e}"),
                )
            })?;

            if bytes.is_empty() {
                continue;
            }

            let saved = state
                .storage
                .save(bytes.as_ref(), file_name.as_deref())
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            stored = Some(saved);
        }

        let stored = stored.ok_or((
            StatusCode::BAD_REQUEST,
            "No file found in multipart upload".to_string(),
        ))?;

        (stored, mime_type)
    } else {
        let (_parts, body) = req.into_parts();
        let body = to_bytes(body, 10 * 1024 * 1024)
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Failed to read body: {e}")))?;

        let parsed: UploadRequest = serde_json::from_slice(&body).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("Expected JSON (or multipart) upload: {e}"),
            )
        })?;

        let file = match parsed {
            UploadRequest::Single(f) => f,
            UploadRequest::Multi(m) => {
                if m.files.len() != 1 {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        "Only one file per request is supported on this endpoint".to_string(),
                    ));
                }
                m.files.into_iter().next().unwrap()
            }
        };

        let bytes = decode_base64_payload(&file.data_base64)?;
        let saved = state
            .storage
            .save(&bytes, file.filename.as_deref())
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        (saved, file.mime_type)
    };

    let url = format!("/api/storage/files/{}", stored.filename);

    Ok(Json(UploadResponse {
        filename: stored.filename,
        mime_type,
        url,
    }))
}

fn is_multipart(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.to_ascii_lowercase().starts_with("multipart/form-data"))
        .unwrap_or(false)
}

fn decode_base64_payload(raw: &str) -> Result<Vec<u8>, (StatusCode, String)> {
    if raw.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Empty base64 payload".to_string()));
    }

    let cleaned = if let Some(idx) = raw.find(',') {
        let (prefix, rest) = raw.split_at(idx + 1);
        if prefix.contains("base64") {
            rest.to_string()
        } else {
            raw.to_string()
        }
    } else {
        raw.to_string()
    };

    STANDARD
        .decode(cleaned.trim())
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid base64: {e}")))
}

pub async fn get_file(
    State(state): State<AppState>,
    Path(filename): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
        return Err((StatusCode::BAD_REQUEST, "Invalid filename".to_string()));
    }

    let path = state.storage.root().join(&filename);
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|_| (StatusCode::NOT_FOUND, "File not found".to_string()))?;

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!("attachment; filename=\"{}\"", filename))
            .unwrap_or_else(|_| HeaderValue::from_static("attachment")),
    );

    Ok((headers, bytes))
}
