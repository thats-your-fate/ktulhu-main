use serde::Deserialize;

use crate::model::message::MessageAttachment;

/// Attachment payload received from the client.
#[derive(Debug, Clone, Deserialize)]
pub struct IncomingAttachment {
    pub id: String,
    pub filename: String,
    #[serde(rename = "mimeType", default)]
    pub mime_type: Option<String>,
    #[serde(rename = "previewBase64", default)]
    pub preview_base64: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "ocrText", default)]
    pub ocr_text: Option<String>,
    #[serde(default)]
    pub labels: Option<Vec<String>>,
}

/// Produce human-readable summaries for attachment content.
pub fn attachment_summaries(attachments: &[IncomingAttachment]) -> Vec<String> {
    attachments
        .iter()
        .map(|att| {
            build_summary(
                att.filename.as_str(),
                att.mime_type.as_deref(),
                att.description.as_deref(),
                att.ocr_text.as_deref(),
                att.labels.as_deref(),
            )
        })
        .collect()
}

/// Summaries derived from stored message attachments.
pub fn message_attachment_summaries(attachments: &[MessageAttachment]) -> Vec<String> {
    attachments
        .iter()
        .map(|att| {
            let label_slice = if att.labels.is_empty() {
                None
            } else {
                Some(att.labels.as_slice())
            };
            build_summary(
                att.filename.as_str(),
                att.mime_type.as_deref(),
                att.description.as_deref(),
                att.ocr_text.as_deref(),
                label_slice,
            )
        })
        .collect()
}

fn build_summary(
    filename: &str,
    mime: Option<&str>,
    description: Option<&str>,
    ocr_text: Option<&str>,
    labels: Option<&[String]>,
) -> String {
    let mut summary = String::new();
    summary.push_str(filename.trim());

    if let Some(mime) = mime.filter(|m| !m.trim().is_empty()) {
        summary.push_str(&format!(" ({})", mime.trim()));
    }

    let detail = ocr_text
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            let snippet = t.trim();
            if snippet.len() > 300 {
                format!("OCR: {}…", &snippet[..300].trim())
            } else {
                format!("OCR: {}", snippet)
            }
        })
        .or_else(|| {
            description
                .filter(|d| !d.trim().is_empty())
                .map(|d| format!("Description: {}", d.trim()))
        })
        .unwrap_or_else(|| "Attachment included for context.".to_string());

    summary.push_str(": ");
    summary.push_str(&detail);

    if let Some(labels) = labels.filter(|labels| !labels.is_empty()) {
        summary.push_str(" Labels: ");
        summary.push_str(&labels.join(", "));
    }

    summary
}
