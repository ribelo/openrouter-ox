use std::marker::PhantomData;

use async_trait::async_trait;
use bon::Builder;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};

use super::traits::{BaseAgent, TypedAgent};
use crate::{tool::ToolBox, OpenRouter};

#[derive(Debug, Clone, Builder)]
pub struct SimpleTypedAgent<T: JsonSchema + Serialize + DeserializeOwned> {
    pub openrouter: OpenRouter,
    #[builder(into)]
    pub name: Option<String>,
    #[builder(into)]
    pub description: Option<String>,
    #[builder(into)]
    pub instructions: Option<String>,
    #[builder(into)]
    pub model: String,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub tools: Option<ToolBox>,
    pub max_iterations: Option<usize>,
    #[builder(default)]
    pub output: PhantomData<T>,
}

#[async_trait]
impl<T> BaseAgent for SimpleTypedAgent<T>
where
    T: Clone + JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn openrouter(&self) -> &OpenRouter {
        &self.openrouter
    }

    fn agent_name(&self) -> &str {
        self.name
            .as_deref()
            .unwrap_or_else(|| std::any::type_name::<Self>().split("::").last().unwrap())
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    fn stop_sequences(&self) -> Option<&Vec<String>> {
        self.stop_sequences.as_ref()
    }

    fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    fn top_p(&self) -> Option<f64> {
        self.top_p
    }

    fn top_k(&self) -> Option<usize> {
        self.top_k
    }

    fn tools(&self) -> Option<&ToolBox> {
        self.tools.as_ref()
    }

    fn max_iterations(&self) -> usize {
        self.max_iterations.unwrap_or(12)
    }
}

// Implement TypedAgent for SimpleTypedAgent
#[async_trait]
impl<T> TypedAgent for SimpleTypedAgent<T>
where
    T: Clone + JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    type Output = T;
}
