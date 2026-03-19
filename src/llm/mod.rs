pub mod anthropic;

use async_trait::async_trait;
use crate::tools::ToolDefinition;
use crate::types::*;

// ---------------------------------------------------------------------------
// Streaming events from LLM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Start of message
    MessageStart,
    /// Incremental text
    TextDelta(String),
    /// Incremental thinking
    ThinkingDelta(String),
    /// A tool call has been fully parsed
    ToolCallStart {
        id: Id,
        name: String,
    },
    /// Incremental tool call arguments (JSON fragment)
    ToolCallDelta {
        id: Id,
        delta: String,
    },
    /// Tool call complete with parsed arguments
    ToolCallEnd {
        id: Id,
        name: String,
        arguments: serde_json::Value,
    },
    /// Message complete
    MessageEnd {
        stop_reason: StopReason,
        usage: Usage,
    },
    /// Error from provider
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Stop,
    ToolUse,
    MaxTokens,
    Error,
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

/// Options for a completion request
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub thinking: ThinkingLevel,
}

/// Trait that LLM providers implement
#[async_trait]
pub trait Provider: Send + Sync {
    /// Provider name (e.g. "anthropic", "openai")
    fn name(&self) -> &str;

    /// Send a completion request, receiving streaming events via callback.
    /// Returns the final assembled assistant message.
    async fn complete(
        &self,
        request: CompletionRequest,
        on_event: &(dyn Fn(StreamEvent) + Send + Sync),
    ) -> anyhow::Result<Message>;
}
