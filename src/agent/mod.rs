use std::{
    fs,
    process::Command,
    sync::{atomic::AtomicBool, Arc},
};

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use crate::inference::llama_cpp_service::LlamaCppService;

#[derive(Debug)]
pub enum Tool {
    RunCmd { cmd: String },
    ReadFile { path: String },
    WriteFile { path: String, content: String },
}

#[derive(Debug)]
pub enum AgentAction {
    Tool { tool: Tool },
    Final { message: String },
}

pub struct AgentState {
    pub history: Vec<String>,
    pub max_steps: usize,
}

pub async fn run_agent(llama: &LlamaCppService, goal: &str) -> Result<()> {
    let cancel = Arc::new(AtomicBool::new(false));
    let mut state = AgentState {
        history: Vec::new(),
        max_steps: 20,
    };

    for step in 0..state.max_steps {
        let prompt = build_prompt(goal, &state);
        let output = llama
            .generate_completion(prompt, cancel.clone())
            .await?;

        let action = parse_action(output.trim())?;

        match action {
            AgentAction::Tool { tool } => {
                let result = execute_tool(tool)?;
                state
                    .history
                    .push(format!("Tool result (step {step}):\n{result}"));
            }
            AgentAction::Final { message } => {
                println!("✅ DONE:\n{message}");
                return Ok(());
            }
        }

        state
            .history
            .push(format!("Model output (step {step}):\n{output}"));
    }

    bail!("agent exceeded max steps");
}

fn build_prompt(goal: &str, state: &AgentState) -> String {
    let mut out = String::new();
    out.push_str("SYSTEM:\n");
    out.push_str("You are a local coding agent.\n");
    out.push_str("Respond ONLY with JSON.\n\n");

    out.push_str("GOAL:\n");
    out.push_str(goal);
    out.push_str("\n\n");

    if !state.history.is_empty() {
        out.push_str("HISTORY:\n");
        for entry in &state.history {
            out.push_str(entry);
            out.push_str("\n\n");
        }
    }

    out.push_str("NEXT ACTION:\n");
    out
}

fn parse_action(text: &str) -> Result<AgentAction> {
    let value: Value = serde_json::from_str(text)?;

    if let Some(final_msg) = value.get("final") {
        let message = final_msg
            .as_str()
            .unwrap_or("")
            .to_string();
        return Ok(AgentAction::Final { message });
    }

    let tool_name = value
        .get("tool")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing tool name"))?;
    let args = value
        .get("args")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("missing args"))?;

    let action = match tool_name {
        "run_cmd" => {
            let cmd = args
                .get("cmd")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing cmd"))?
                .to_string();
            Tool::RunCmd { cmd }
        }
        "read_file" => {
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing path"))?
                .to_string();
            Tool::ReadFile { path }
        }
        "write_file" => {
            let path = args
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing path"))?
                .to_string();
            let content = args
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing content"))?
                .to_string();
            Tool::WriteFile { path, content }
        }
        other => bail!("unknown tool: {other}"),
    };

    Ok(AgentAction::Tool { tool: action })
}

fn execute_tool(tool: Tool) -> Result<String> {
    match tool {
        Tool::RunCmd { cmd } => {
            let output = Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .output()?;
            Ok(format!(
                "status: {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ))
        }
        Tool::ReadFile { path } => {
            let contents = fs::read_to_string(&path)?;
            Ok(format!("Read {} bytes from {}", contents.len(), path))
        }
        Tool::WriteFile { path, content } => {
            fs::write(&path, content)?;
            Ok(format!("Wrote file {}", path))
        }
    }
}
