use anyhow::Result;
use std::{path::PathBuf, sync::Arc};

use crate::inference::{mistral::MistralService, phi::PhiService, roberta::RobertaService};

pub struct ModelManager {
    pub mistral: Arc<MistralService>, // GPU0
    pub phi: Arc<PhiService>,         // GPU1
    pub roberta: Arc<RobertaService>, // GPU1 (classifier)
}

impl ModelManager {
    pub async fn new() -> Result<Self> {
        let mistral_dir = PathBuf::from("models/ministral_8b");
        let phi_dir = PathBuf::from("models/phi3mini");
        let roberta_dir = PathBuf::from("models/roberta");

        let mistral = Arc::new(MistralService::new_with(mistral_dir, 0).await?);
        let phi = Arc::new(PhiService::new_with(phi_dir, 1).await?);
        let roberta = Arc::new(RobertaService::new_with(roberta_dir, 1).await?);

        Ok(Self {
            mistral,
            phi,
            roberta,
        })
    }
}
