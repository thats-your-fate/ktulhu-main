use axum::{extract::State, http::StatusCode, Json};
use std::sync::Arc;

use crate::ws::AppState;
use super::dto::{EmbedRequest, EmbedResponse};
use super::service::RobertaService;

pub async fn embed_handler(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    run_embed_job(state.models.roberta.clone(), req).await
}

pub async fn embed_handler_3060(
    State(state): State<AppState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    run_embed_job(state.models.roberta_embed.clone(), req).await
}

async fn run_embed_job(
    service: Arc<RobertaService>,
    req: EmbedRequest,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    if req.texts.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "texts_required".into()));
    }

    let mut embeddings = Vec::with_capacity(req.texts.len());
    let mut dims = 0;

    for text in &req.texts {
        let e = service
            .embed(text)
            .await
            .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
        dims = e.len();
        embeddings.push(e);
    }

    Ok(Json(EmbedResponse {
        model: "roberta-base".into(),
        dims,
        embeddings,
    }))
}
