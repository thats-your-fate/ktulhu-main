use anyhow::Result;

use crate::{manager::ModelManager, prompts};

const INTENT_CONFIDENCE_THRESHOLD: f32 = 0.65;
const CLARIFICATION_THRESHOLD: f32 = 0.5;
const MULTI_INTENT_MIN_LEN: usize = 40;

#[derive(Debug, Clone)]
pub struct IntentRoutingResult {
    pub intent: String,
    pub confidence: f32,
    pub multi_intent: bool,
    pub clarification_needed: bool,
    pub notes: Vec<String>,
}

impl Default for IntentRoutingResult {
    fn default() -> Self {
        Self {
            intent: prompts::default_intent().to_string(),
            confidence: 0.0,
            multi_intent: false,
            clarification_needed: false,
            notes: Vec::new(),
        }
    }
}

pub async fn route_intent(models: &ModelManager, text: &str) -> Result<IntentRoutingResult> {
    let utterances = split_into_utterances(text);
    let multi_intent = utterances
        .iter()
        .filter(|segment| segment.trim().chars().count() >= MULTI_INTENT_MIN_LEN)
        .count()
        > 1;

    if multi_intent {
        let mut result = IntentRoutingResult::default();
        result.multi_intent = true;
        result
            .notes
            .push("multiple significant utterances detected".to_string());
        return Ok(result);
    }

    let classification_target = utterances
        .first()
        .cloned()
        .unwrap_or_else(|| text.to_string());

    let cleaned_target = clean_intent_text(&classification_target);
    let classify_input = if cleaned_target.trim().is_empty() {
        classification_target.as_str()
    } else {
        cleaned_target.as_str()
    };

    let (mut intent, mut confidence) =
        crate::classifier::intent::classify_intent(models, classify_input).await?;

    let mut notes = Vec::new();
    let lower_text = text.to_lowercase();

    if let Some(override_intent) = apply_intent_rules(text, intent.as_str()) {
        if override_intent != intent {
            notes.push(format!(
                "rule override: {} → {}",
                intent.as_str(),
                override_intent.as_str()
            ));
            intent = override_intent;
        }
    }

    if starts_with_task_command(&lower_text) && intent != "task_short" {
        notes.push("hard task override".to_string());
        intent = "task_short".to_string();
    }

    if intent == "chat_casual" && !is_chat_casual_allowed(text) {
        let forced = select_fallback_intent(text, intent.as_str());
        if forced != intent {
            notes.push(format!(
                "chat_casual disallowed; forcing {}",
                forced.as_str()
            ));
            intent = forced;
        }
    }

    if confidence < INTENT_CONFIDENCE_THRESHOLD {
        let fallback = select_fallback_intent(text, intent.as_str());
        if fallback != intent {
            notes.push(format!(
                "confidence {:.2} below threshold; fallback {} → {}",
                confidence,
                intent.as_str(),
                fallback.as_str()
            ));
            intent = fallback;
        } else {
            notes.push(format!(
                "confidence {:.2} below threshold; keeping intent {}",
                confidence,
                intent.as_str()
            ));
        }
    }

    let clarification_needed = confidence < CLARIFICATION_THRESHOLD && notes.is_empty();

    Ok(IntentRoutingResult {
        intent,
        confidence,
        multi_intent: false,
        clarification_needed,
        notes,
    })
}

fn split_into_utterances(text: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut buffer = String::new();
    let mut newline_streak = 0usize;

    for ch in text.chars() {
        buffer.push(ch);

        if ch == '\n' {
            newline_streak += 1;
            if newline_streak >= 2 {
                push_segment(&mut buffer, &mut segments);
                newline_streak = 0;
            }
            continue;
        }

        newline_streak = 0;

        if ch == '?' || ch == '!' {
            push_segment(&mut buffer, &mut segments);
        }
    }

    push_segment(&mut buffer, &mut segments);

    if segments.is_empty() && !text.trim().is_empty() {
        segments.push(text.trim().to_string());
    }

    segments
}

fn push_segment(buffer: &mut String, segments: &mut Vec<String>) {
    let trimmed = buffer.trim();
    if !trimmed.is_empty() {
        segments.push(trimmed.to_string());
    }
    buffer.clear();
}

fn clean_intent_text(text: &str) -> String {
    let lower = text.trim().to_lowercase();
    let mut tokens: Vec<&str> = lower.split_whitespace().collect();

    while let Some(token) = tokens.first() {
        if is_casual_marker(token) {
            tokens.remove(0);
        } else {
            break;
        }
    }

    let recombined = tokens.join(" ");
    strip_emojis(&recombined).trim().to_string()
}

fn is_casual_marker(token: &str) -> bool {
    matches!(
        token,
        "lol" | "haha" | "lmao" | "hey" | "hi" | "pls" | "please" | "omg"
    )
}

fn strip_emojis(text: &str) -> String {
    text.chars().filter(|c| !is_emoji(*c)).collect::<String>()
}

fn is_emoji(c: char) -> bool {
    matches!(
        c,
        '\u{1F300}'..='\u{1FAFF}'
            | '\u{1F1E6}'..='\u{1F1FF}'
            | '\u{1F900}'..='\u{1F9FF}'
            | '\u{2600}'..='\u{27BF}'
            | '\u{1F000}'..='\u{1F02F}'
    )
}

fn apply_intent_rules(text: &str, current_intent: &str) -> Option<String> {
    let lower = text.to_lowercase();

    if contains_culture_keywords(&lower) {
        return Some("culture_context".to_string());
    }

    if starts_with_task_command(&lower) {
        return Some("task_short".to_string());
    }

    if contains_advice_patterns(&lower) {
        return Some("advice_practical".to_string());
    }

    if current_intent == "opinion_reflective" && lower.contains("what do you think") {
        return Some("opinion_reflective".to_string());
    }

    None
}

fn select_fallback_intent(text: &str, current_intent: &str) -> String {
    let lower = text.to_lowercase();

    if starts_with_task_command(&lower) {
        return "task_short".to_string();
    }

    if contains_advice_patterns(&lower) {
        return "advice_practical".to_string();
    }

    if current_intent == "task_short"
        || current_intent == "advice_practical"
        || current_intent == "opinion_reflective"
    {
        return current_intent.to_string();
    }

    if has_emotional_language(&lower) && is_chat_casual_allowed(text) {
        return "chat_casual".to_string();
    }

    "task_short".to_string()
}

fn contains_culture_keywords(text: &str) -> bool {
    const CULTURE_KEYWORDS: &[&str] = &[
        // Spanish
"cultura",
"cultural",
"tradición",
"tradiciones",
"costumbre",
"costumbres",
"ritual",
"rituales",
"país",
"pais",
"idioma",
"lengua",
"regional",
"región",
"religión",
"religion",
"étnico",
"etnia",
"herencia",
"visitar",
"visitando",
"viajar",
"viaje",
"turista",
"turismo",
"extranjero",
"extranjera",
"afuera",
"etiqueta",
"educado",
"grosero",
"descortés",
"respetuoso",

        "culture",
        "cultural",
        "tradition",
        "custom",
        "customs",
        "ritual",
        "country",
        "language",
        "regional",
        "religion",
        "ethnic",
        "heritage",
        "visit",
        "visiting",
        "travel",
        "traveling",
        "tourist",
        "abroad",
        "foreign",
        "etiquette",
        "rude",
        "polite",
    ];

    CULTURE_KEYWORDS.iter().any(|kw| text.contains(kw))
}

fn starts_with_task_command(text: &str) -> bool {
    let trimmed = text.trim_start();
    const PREFIXES: &[&str] = &[
        "write ",
        "draft ",
        "compose ",
        "summarize ",
        "generate ",
        "create ",
        "message ",
        "send a message",
        // Spanish prefixes
"escribe ",
"redacta ",
"redacte ",
"crea ",
"cree ",
"genera ",
"genere ",
"resume ",
"resuma ",
"haz ",
"haga ",
"envía ",
"envíe ",
"manda ",
"mande ",

    ];

    if PREFIXES.iter().any(|prefix| trimmed.starts_with(prefix)) {
        return true;
    }

    const KEYWORDS: &[&str] = &[
        " write ",
        " draft ",
        " compose ",
        " summarize ",
        " generate ",
        " create ",
        " write a message",
        " draft a message",
        " compose a message",
        " send a message",
        " message for ",
        " message to ",
        // Spanish keywords
" escribe ",
" redacta ",
" redacte ",
" crea ",
" genere ",
" resume ",
" mensaje para ",
" mensaje a ",
" escribir un mensaje",
" redactar un mensaje",
" crear un mensaje",
" enviar un mensaje",

    ];

    KEYWORDS.iter().any(|kw| text.contains(kw))
}

fn contains_advice_patterns(text: &str) -> bool {
    let phrases = [
        "how can i",
        "how do i",
        "how to",
        "ways to",
        "what should i do",
        "best way to",
        "tips for",
        "advice on",
        "help me",
        "improve my",
        // Spanish
"cómo puedo",
"como puedo",
"cómo hago",
"como hago",
"qué puedo hacer",
"que puedo hacer",
"formas de",
"mejores formas de",
"mejor manera de",
"consejos para",
"recomendaciones para",
"ayúdame a",
"ayudame a",
"quiero mejorar",
"mejorar mi",

    ];
    phrases.iter().any(|phrase| text.contains(phrase))
}

fn has_emotional_language(text: &str) -> bool {
    const EMOTIONAL_KEYWORDS: &[&str] = &[
        "sad",
        "happy",
        "excited",
        "upset",
        "frustrated",
        "worried",
        "stressed",
        "angry",
        "afraid",
        "thrilled",
        "depressed",
        "anxious",
        "lonely",
        "love",
        "hate",
        "tired",
        "exhausted",
        // Spanish
"triste",
"feliz",
"emocionado",
"emocionada",
"contento",
"contenta",
"molesto",
"molesta",
"frustrado",
"frustrada",
"preocupado",
"preocupada",
"estresado",
"estresada",
"enojado",
"enojada",
"asustado",
"asustada",
"ansioso",
"ansiosa",
"solo",
"sola",
"cansado",
"cansada",
"agotado",
"agotada",
"abrumado",
"abrumada",

    ];

    EMOTIONAL_KEYWORDS.iter().any(|kw| text.contains(kw))
}

fn is_chat_casual_allowed(text: &str) -> bool {
    let lower = text.to_lowercase();
    has_emotional_language(&lower)
        && !contains_culture_keywords(&lower)
        && !contains_advice_patterns(&lower)
        && !starts_with_task_command(&lower)
}
