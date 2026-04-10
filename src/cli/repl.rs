//! REPL Module - Interactive Read-Eval-Print Loop
//!
//! Beautiful REPL interface matching the original Claude Code aesthetic.
//! Full async tool-call loop support.

use crate::api::{ApiClient, ChatMessage, ToolDefinition};
use crate::cli::ui;
use crate::state::AppState;
use crate::tools::{ToolOutput, ToolRegistry};
use colored::Colorize;
use std::io::{self, BufRead, Write};
use tokio::task::spawn_blocking;

// ─────────────────────────────────────────────────────────────────────────────
// Repr struct — now carries a ToolRegistry so tools are available in-process.
// ─────────────────────────────────────────────────────────────────────────────

pub struct Repl {
    state: AppState,
    conversation_history: Vec<ChatMessage>,
    tool_registry: ToolRegistry,
}

impl Repl {
    pub fn new(state: AppState) -> Self {
        ui::init_terminal();
        Self {
            state,
            conversation_history: Vec::new(),
            tool_registry: ToolRegistry::new(),
        }
    }

    // ── Public entry point (async) ──────────────────────────────────────────

    pub async fn start(&mut self, initial_prompt: Option<String>) -> anyhow::Result<()> {
        ui::print_welcome();

        if let Some(prompt) = initial_prompt {
            self.process_input(&prompt).await?;
        }

        // We need a mutable stdout for the prompt loop; take it once.
        let mut stdout = io::stdout();

        loop {
            ui::print_prompt();
            stdout.flush()?;

            // stdin.read_line is blocking — run it on a blocking thread so the
            // async runtime stays free.
            let input = spawn_blocking(|| {
                let stdin = io::stdin();
                let mut buf = String::new();
                stdin.lock().read_line(&mut buf).map(|_| buf.trim().to_string())
            })
            .await??;

            if input.is_empty() {
                continue;
            }

            match input.as_str() {
                "exit" | "quit" | ".exit" | ":q" => {
                    println!(
                        "\n  {} {}\n",
                        "👋".yellow(),
                        "Goodbye!".truecolor(255, 140, 66).bold()
                    );
                    break;
                }
                "help" | ".help" | ":h" => ui::print_help(),
                "status" | ".status" => self.print_status(),
                "clear" | ".clear" | ":c" => ui::clear_screen(),
                "history" | ".history" => self.print_history(),
                "reset" | ".reset" => self.reset_conversation(),
                "config" | ".config" => self.print_config(),
                _ => {
                    if let Err(e) = self.process_input(&input).await {
                        ui::print_error(&format!("Error: {}", e));
                    }
                }
            }
        }

        Ok(())
    }

    // ── Core conversation + tool-call loop ───────────────────────────────────

    /// Send a user message, call the API, handle tool calls, print response.
    async fn process_input(&mut self, input: &str) -> anyhow::Result<()> {
        ui::print_user_message(input);

        let client = ApiClient::new(self.state.settings.clone());

        // Check API key is present
        if client.get_api_key().is_none() {
            ui::print_error(
                "API key not configured\n\nSet it with:\n  claude-code config set api_key \"your-api-key\"",
            );
            return Ok(());
        }

        self.conversation_history.push(ChatMessage::user(input));

        // Build tool definitions from the registry
        let tools = Some(self.build_tool_definitions());

        // Keep calling the API until we get a text response (no tool_calls).
        let mut loop_count = 0;
        const MAX_LOOPS: usize = 10; // safety guard

        loop {
            loop_count += 1;
            if loop_count > MAX_LOOPS {
                ui::print_error("Too many tool-call loops — aborting to prevent infinite recursion.");
                break;
            }

            ui::print_typing_indicator();

            let messages = self.conversation_history.clone();

            // ── First API call ─────────────────────────────────────────────
            let response = client
                .chat(messages, tools.clone())
                .await
                .map_err(|e| anyhow::anyhow!("API error: {}", e))?;

            let choice = match response.choices.first() {
                Some(c) => c,
                None => {
                    ui::print_error("Empty response from API");
                    break;
                }
            };

            let finish_reason = choice.finish_reason.clone().unwrap_or_default();

            // ── Text response — we're done ─────────────────────────────────
            if finish_reason == "stop" || finish_reason == "length" {
                if let Some(content) = &choice.message.content {
                    if !content.is_empty() {
                        ui::print_claude_message(content);
                        self.conversation_history
                            .push(ChatMessage::assistant(content.clone()));
                    }
                }
                self.print_usage(&response.usage);
                break;
            }

            // ── Tool calls — execute them and loop ─────────────────────────
            let tool_calls = choice.message.tool_calls.clone().unwrap_or_default();

            if tool_calls.is_empty() {
                // No content, no tools — just break
                break;
            }

            ui::print_tool_calls_start();

            // Collect tool results
            let mut tool_results: Vec<ChatMessage> = Vec::with_capacity(tool_calls.len());

            for tc in &tool_calls {
                let name = &tc.function.name;
                let args_str = &tc.function.arguments;

                ui::print_tool_call(name, args_str);

                // Parse arguments JSON
                let args: serde_json::Value = args_str
                    .parse()
                    .unwrap_or(serde_json::Value::Null);

                // Execute via registry
                let output: ToolOutput = match self.tool_registry.execute(name, args).await {
                    Ok(out) => out,
                    Err(e) => ToolOutput {
                        output_type: "error".to_string(),
                        content: e.message,
                        metadata: Default::default(),
                    },
                };

                let is_error = output.output_type == "error";
                ui::print_tool_result(name, &output.content, is_error);

                tool_results.push(ChatMessage::tool(&tc.id, &output.content));
            }

            ui::print_tool_calls_end();

            // Add assistant message with tool_calls and all tool result messages
            self.conversation_history
                .push(ChatMessage::assistant_with_tools(tool_calls));
            self.conversation_history.extend(tool_results);

            // Loop again — the API will see the tool results and respond.
        }

        Ok(())
    }

    // ── Tool definitions ────────────────────────────────────────────────────

    /// Convert every registered tool into an OpenAI-compatible ToolDefinition.
    fn build_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tool_registry
            .list()
            .into_iter()
            .map(|t| {
                let json = t.tool_definition();
                // json is a serde_json::Value — map to our ToolDefinition struct
                let obj = json.as_object().expect("tool_definition must be an object");
                let func = obj.get("function").expect("function key missing");
                let func_obj = func.as_object().expect("function must be an object");
                ToolDefinition {
                    r#type: obj.get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("function")
                        .to_string(),
                    function: crate::api::ToolFunction {
                        name: func_obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        description: func_obj
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        parameters: func_obj
                            .get("parameters")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                    },
                }
            })
            .collect()
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn print_usage(&self, usage: &Option<crate::api::Usage>) {
        if let Some(u) = usage {
            let total = u.prompt_tokens + u.completion_tokens;
            println!(
                "  {} {} prompt · {} generated · {} total",
                "◦".truecolor(100, 100, 100),
                u.prompt_tokens.to_string().truecolor(150, 150, 150),
                u.completion_tokens.to_string().truecolor(150, 150, 150),
                total.to_string().truecolor(180, 180, 180)
            );
        }
    }

    fn print_status(&self) {
        let status = ui::StatusInfo {
            model: self.state.settings.model.clone(),
            api_base: self.state.settings.api.base_url.clone(),
            max_tokens: self.state.settings.api.max_tokens.to_string(),
            timeout: self.state.settings.api.timeout,
            streaming: self.state.settings.api.streaming,
            message_count: self.conversation_history.len(),
            api_key_set: self.state.settings.api.get_api_key().is_some(),
        };
        ui::print_status(&status);
    }

    fn print_history(&self) {
        println!();
        if self.conversation_history.is_empty() {
            println!(
                "  {} {}",
                "◦".truecolor(100, 100, 100),
                "No conversation history".bright_black()
            );
        } else {
            println!(
                "  {} {}",
                "◦".truecolor(147, 112, 219),
                format!(
                    "Conversation history ({} messages)",
                    self.conversation_history.len()
                )
                .truecolor(147, 112, 219)
                .bold()
            );
            println!();

            for (i, msg) in self.conversation_history.iter().enumerate() {
                let role_label = match msg.role.as_str() {
                    "user" => "You".truecolor(255, 180, 100),
                    "assistant" => "Claude".truecolor(200, 150, 255),
                    "tool" => "Tool".truecolor(80, 200, 255),
                    _ => "System".bright_black(),
                };

                let content = msg.content.as_deref().unwrap_or("");
                let preview: String = content.chars().take(50).collect();
                let suffix = if content.len() > 50 {
                    "..."
                } else {
                    ""
                };

                println!(
                    "  {}. {}  {}{}",
                    (i + 1).to_string().truecolor(100, 100, 100),
                    role_label,
                    preview.bright_white(),
                    suffix.bright_black()
                );
            }
        }
        println!();
    }

    fn print_config(&self) {
        println!();
        println!(
            "  {} {}",
            "⚙".truecolor(147, 112, 219),
            "Configuration".truecolor(147, 112, 219).bold()
        );
        println!();

        match serde_json::to_string_pretty(&self.state.settings) {
            Ok(json) => {
                for line in json.lines() {
                    println!("  {}", line.bright_white());
                }
            }
            Err(_) => {
                ui::print_error("Failed to serialize configuration");
            }
        }
        println!();
    }

    fn reset_conversation(&mut self) {
        self.conversation_history.clear();
        ui::print_success("Conversation reset");
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_repl_creation() {
        let state = AppState::default();
        let repl = Repl::new(state);
        assert!(repl.conversation_history.is_empty());
    }

    #[tokio::test]
    async fn test_tool_registry_has_tools() {
        let state = AppState::default();
        let repl = Repl::new(state);
        let defs = repl.build_tool_definitions();
        // We registered 9 built-in tools
        assert!(!defs.is_empty(), "should have at least one tool");
        let names: Vec<_> = defs.iter().map(|d| d.function.name.clone()).collect();
        assert!(names.contains(&"file_read".to_string()), "should have file_read");
        assert!(names.contains(&"execute_command".to_string()), "should have execute_command");
    }
}
