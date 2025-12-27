use once_cell::sync::Lazy;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct ByteStreamDecoder {
    pending_bytes: Vec<u8>,
}

static BYTE_DECODER: Lazy<HashMap<char, u8>> = Lazy::new(build_byte_decoder);

pub fn decode_byte_fallback<'a>(text: &'a str) -> Cow<'a, str> {
    if text.is_empty() {
        return Cow::Borrowed(text);
    }

    let mut bytes = Vec::with_capacity(text.len());
    let mut mapped = false;

    for ch in text.chars() {
        if let Some(byte) = BYTE_DECODER.get(&ch) {
            bytes.push(*byte);
            mapped = true;
        } else {
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            bytes.extend_from_slice(encoded.as_bytes());
        }
    }

    if !mapped {
        return Cow::Borrowed(text);
    }

    match String::from_utf8(bytes) {
        Ok(s) => Cow::Owned(s),
        Err(_) => Cow::Borrowed(text),
    }
}

pub fn normalize_token_text(text: &str) -> String {
    let decoded = decode_byte_fallback(text);
    tidy_decoded_text(decoded.as_ref())
}

impl ByteStreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &str) -> Option<String> {
        if chunk.is_empty() {
            return None;
        }

        append_decoded_bytes(&mut self.pending_bytes, chunk);
        self.take_ready_text()
    }

    pub fn flush(&mut self) -> Option<String> {
        self.take_ready_text()
    }

    pub fn is_empty(&self) -> bool {
        self.pending_bytes.is_empty()
    }

    fn take_ready_text(&mut self) -> Option<String> {
        if self.pending_bytes.is_empty() {
            return None;
        }

        match std::str::from_utf8(&self.pending_bytes) {
            Ok(valid) => {
                let text = tidy_decoded_text(valid);
                self.pending_bytes.clear();
                if text.is_empty() {
                    None
                } else {
                    Some(text)
                }
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to == 0 {
                    return None;
                }

                let valid =
                    unsafe { std::str::from_utf8_unchecked(&self.pending_bytes[..valid_up_to]) };
                let text = tidy_decoded_text(valid);
                self.pending_bytes.drain(0..valid_up_to);
                if text.is_empty() {
                    None
                } else {
                    Some(text)
                }
            }
        }
    }
}

fn append_decoded_bytes(target: &mut Vec<u8>, chunk: &str) {
    for ch in chunk.chars() {
        if let Some(byte) = BYTE_DECODER.get(&ch) {
            target.push(*byte);
        } else {
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            target.extend_from_slice(encoded.as_bytes());
        }
    }
}

fn tidy_decoded_text(text: &str) -> String {
    text.replace('\u{0120}', " ")
        .replace('\u{2581}', " ")
        .replace('\u{200b}', "")
}

#[cfg(test)]
mod tests {
    use super::{normalize_token_text, ByteStreamDecoder};

    #[test]
    fn decodes_emoji_from_byte_fallback() {
        assert_eq!(normalize_token_text("ðŁĺĬ"), "😊");
    }

    #[test]
    fn decodes_multiple_emojis() {
        assert_eq!(normalize_token_text("ðŁĻĤ ðŁĺĬ"), "🙂 😊");
    }

    #[test]
    fn stream_decoder_waits_for_full_emoji() {
        let mut decoder = ByteStreamDecoder::new();
        assert!(decoder.push("ð").is_none());
        assert!(decoder.push("Ł").is_none());
        assert!(decoder.push("ĺ").is_none());
        assert_eq!(decoder.push("Ĭ"), Some("😊".to_string()));
        assert!(decoder.is_empty());
    }

    #[test]
    fn stream_decoder_handles_plain_text() {
        let mut decoder = ByteStreamDecoder::new();
        assert_eq!(decoder.push("Hello"), Some("Hello".into()));
        assert!(decoder.push(" ").is_some());
    }
}

fn build_byte_decoder() -> HashMap<char, u8> {
    let mut bs = Vec::new();
    let mut cs = Vec::new();

    for b in 33..=126 {
        bs.push(b as u32);
        cs.push(b as u32);
    }

    for b in 161..=172 {
        bs.push(b as u32);
        cs.push(b as u32);
    }

    for b in 174..=255 {
        bs.push(b as u32);
        cs.push(b as u32);
    }

    let mut seen: HashSet<u32> = bs.iter().copied().collect();
    let mut n = 0u32;
    for b in 0u32..=255 {
        if !seen.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            seen.insert(b);
            n += 1;
        }
    }

    let mut decoder = HashMap::new();
    for (encoded, byte) in cs.into_iter().zip(bs.into_iter()) {
        if let (Some(ch), Ok(b)) = (char::from_u32(encoded), u8::try_from(byte)) {
            decoder.insert(ch, b);
        }
    }

    decoder
}
