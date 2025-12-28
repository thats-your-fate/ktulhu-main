use once_cell::sync::OnceCell;
use serde::Deserialize;
use tokio::sync::RwLock;

#[derive(Debug, Deserialize, Clone)]
pub struct Jwk {
    pub kid: String,
    pub n: String,
    pub e: String,
}

#[derive(Debug, Deserialize)]
pub struct JwkSet {
    pub keys: Vec<Jwk>,
}

/// Singleton cache for Google JWKS
pub struct GoogleJwkCache {
    inner: RwLock<Vec<Jwk>>,
}

static INSTANCE: OnceCell<GoogleJwkCache> = OnceCell::new();

impl GoogleJwkCache {
    pub fn instance() -> &'static GoogleJwkCache {
        INSTANCE.get_or_init(|| GoogleJwkCache {
            inner: RwLock::new(Vec::new()),
        })
    }

    pub async fn get_key(&self, kid: &str) -> anyhow::Result<Jwk> {
        {
            // 1) try local cache
            let keys = self.inner.read().await;
            if let Some(k) = keys.iter().find(|k| k.kid == kid) {
                return Ok(k.clone());
            }
        }

        // 2) fetch JWKS
        let fetched: JwkSet = reqwest::get("https://www.googleapis.com/oauth2/v3/certs")
            .await?
            .json()
            .await?;

        {
            // store to cache
            let mut keys = self.inner.write().await;
            *keys = fetched.keys.clone();
        }

        // 3) return matching key
        let keys = self.inner.read().await;
        keys.iter()
            .find(|k| k.kid == kid)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Google key not found"))
    }
}
