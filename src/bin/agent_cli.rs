use anyhow::Result;
use ktulhuMain::{agent, manager::ModelManager};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    dotenvy::from_filename("config/llamacpp.env").ok();

    let goal = env::args().skip(1).collect::<Vec<_>>().join(" ");
    let goal = if goal.trim().is_empty() {
        "Run tests and fix any failures".to_string()
    } else {
        goal
    };

    println!("🎯 Agent goal: {goal}");

    let models = ModelManager::new().await?;
    agent::run_agent(models.mistral_llama.as_ref(), &goal).await
}
