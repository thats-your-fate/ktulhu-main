use serde::Deserialize;

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
            let mut summary = String::new();
            summary.push_str(att.filename.trim());

            if let Some(mime) = att.mime_type.as_deref().filter(|m| !m.trim().is_empty()) {
                summary.push_str(&format!(" ({})", mime.trim()));
            }

            let detail = att
                .ocr_text
                .as_deref()
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
                    att.description
                        .as_deref()
                        .filter(|d| !d.trim().is_empty())
                        .map(|d| format!("Description: {}", d.trim()))
                })
                .unwrap_or_else(|| "Attachment included for context.".to_string());

            summary.push_str(": ");
            summary.push_str(&detail);

            if let Some(labels) = att.labels.as_ref().filter(|l| !l.is_empty()) {
                summary.push_str(" Labels: ");
                summary.push_str(&labels.join(", "));
            }

            summary
        })
        .collect()
}
