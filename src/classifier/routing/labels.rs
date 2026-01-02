use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct RoutingLabelFile {
    layer1: HashMap<String, String>,
    task_types: HashMap<String, String>,
}

pub struct RoutingLabelSet {
    layer1: HashMap<String, String>,
    task_types: HashMap<String, String>,
}

impl RoutingLabelSet {
    pub fn layer1_display(&self, key: &str) -> String {
        self.layer1
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string())
    }

    pub fn task_display(&self, key: &str) -> String {
        self.task_types
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.to_string())
    }
}

macro_rules! routing_labels_file {
    ($lang:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/lang/",
            $lang,
            "/routing_labels.json"
        ))
    };
}

static EN_ROUTING_LABELS: Lazy<RoutingLabelSet> =
    Lazy::new(|| load_routing_labels(routing_labels_file!("en")));
static ES_ROUTING_LABELS: Lazy<RoutingLabelSet> =
    Lazy::new(|| load_routing_labels(routing_labels_file!("es")));
static RU_ROUTING_LABELS: Lazy<RoutingLabelSet> =
    Lazy::new(|| load_routing_labels(routing_labels_file!("ru")));
static PT_ROUTING_LABELS: Lazy<RoutingLabelSet> =
    Lazy::new(|| load_routing_labels(routing_labels_file!("pt")));

pub fn routing_labels(language: Option<&str>) -> &'static RoutingLabelSet {
    let normalized = language
        .and_then(|lang| lang.split(|c| c == '-' || c == '_').next())
        .unwrap_or("en")
        .to_ascii_lowercase();

    match normalized.as_str() {
        "es" => &ES_ROUTING_LABELS,
        "ru" => &RU_ROUTING_LABELS,
        "pt" => &PT_ROUTING_LABELS,
        _ => &EN_ROUTING_LABELS,
    }
}

fn load_routing_labels(raw: &str) -> RoutingLabelSet {
    let parsed: RoutingLabelFile = serde_json::from_str(raw).expect("invalid routing label config");
    RoutingLabelSet {
        layer1: parsed.layer1,
        task_types: parsed.task_types,
    }
}
