use anyhow::Result;
use std::{path::PathBuf, sync::Arc};

use crate::inference::{
    ministral8_b::Ministral8BService, ministral_8b_resoning::Ministral8BResoningService,
    phi::PhiService, roberta::RobertaService,
};

pub struct ModelManager {
    pub mistral_reasoning: Arc<Ministral8BResoningService>, // GPU0
    pub mistral8_b: Arc<Ministral8BService>,               // GPU0 (legacy)
    pub phi: Arc<PhiService>,                              // GPU1
    pub roberta: Arc<RobertaService>,                      // GPU0 (classifier)
}

impl ModelManager {
    pub async fn new() -> Result<Self> {
        let mistral_reasoning_dir = PathBuf::from("models/ministral_8b_resoning");
        let mistral8_b_dir = PathBuf::from("models/ministral8_b");
        let phi_dir = PathBuf::from("models/phi3mini");
        let roberta_dir = PathBuf::from("models/fbrobertabig");

        let mistral_reasoning =
            Arc::new(Ministral8BResoningService::new_with(mistral_reasoning_dir, 0).await?);
        let mistral8_b = Arc::new(Ministral8BService::new_with(mistral8_b_dir, 0).await?);
        let phi = Arc::new(PhiService::new_with(phi_dir, 1).await?);
        let roberta = Arc::new(RobertaService::new_with(roberta_dir, 0).await?);

        Ok(Self {
            mistral_reasoning,
            mistral8_b,
            phi,
            roberta,
        })
    }
}
