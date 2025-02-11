use std::{
    collections::HashMap,
    fmt::{self, Debug},
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use bon::Builder;
use schemars::{schema_for, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

use crate::{message::ToolMessage, response::ToolCall};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDescription {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub name: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDescription,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "auto")]
    Auto,
    Function {
        #[serde(rename = "type")]
        choice_type: String,
        function: FunctionName,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionName {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadata {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Clone, Default, Builder)]
pub struct ToolBox {
    #[builder(field)]
    tools: Arc<RwLock<HashMap<String, Arc<dyn AnyTool>>>>,
}

impl<S: tool_box_builder::State> ToolBoxBuilder<S> {
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        let name = tool.name().to_string();
        self.tools.write().unwrap().insert(name, Arc::new(tool));
        self
    }
}

impl fmt::Debug for ToolBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tools = self.tools.read().map_err(|_| fmt::Error)?;
        f.debug_struct("ToolBox")
            .field("tools", &format!("HashMap with {} entries", tools.len()))
            .finish()
    }
}

#[derive(Debug, thiserror::Error, Serialize)]
pub enum ToolError {
    #[error("Failed to execute tool: {0}")]
    ExecutionFailed(String),
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Failed to deserialize input: {0}")]
    InputDeserializationFailed(String),
    #[error("Failed to serialize output: {0}")]
    OutputSerializationFailed(String),
    #[error("Failed to generate input schema: {0}")]
    SchemaGenerationFailed(String),
    #[error("Missing arguments: {0}")]
    MissingArguments(String),
}

impl ToolBox {
    pub fn add<T: Tool + 'static>(&self, tool: T) {
        let name = tool.name().to_string();
        self.tools.write().unwrap().insert(name, Arc::new(tool));
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn AnyTool>> {
        self.tools.read().unwrap().get(name).cloned()
    }

    pub async fn invoke(&self, tool_call: ToolCall) -> ToolMessage {
        match self.get(&tool_call.function.name) {
            Some(tool) => tool.invoke_any(tool_call).await,
            None => ToolMessage {
                content: ToolError::ToolNotFound(tool_call.function.name).to_string(),
                tool_call_id: tool_call.id,
            },
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.read().unwrap().is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.read().unwrap().len()
    }

    #[must_use]
    pub fn metadata(&self) -> Vec<ToolMetadata> {
        self.tools
            .read()
            .unwrap()
            .values()
            .map(|tool| ToolMetadata {
                tool_type: "function".to_string(),
                function: FunctionMetadata {
                    name: tool.name().to_string(),
                    description: tool.description().map(|s| s.to_string()),
                    parameters: tool.input_schema(),
                },
            })
            .collect()
    }
}

impl Serialize for ToolBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        self.metadata().serialize(serializer)
    }
}

#[async_trait]
pub trait AnyTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str>;
    async fn invoke_any(&self, tool_call: ToolCall) -> ToolMessage;
    fn input_schema(&self) -> Value;
}

#[async_trait]
pub trait Tool: Clone + Send + Sync {
    type Input: JsonSchema + DeserializeOwned + Debug + Send + Sync;
    type Output: Serialize + Debug + Send + Sync;
    type Error: ToString + Debug;
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str> {
        None
    }
    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    fn tool_schema(&self) -> Value {
        let json_schema = schema_for!(Self::Input);
        let mut input_schema = serde_json::to_value(json_schema).unwrap();
        input_schema
            .as_object_mut()
            .unwrap()
            .remove("title")
            .unwrap();
        if input_schema.get("properties").is_some() {
            input_schema
        } else {
            serde_json::json!(None::<()>)
        }
    }
    fn direct_output(&self) -> bool {
        false
    }
}

#[async_trait]
impl<T: Tool + Send + Sync> AnyTool for T {
    fn name(&self) -> &str {
        self.name()
    }

    fn description(&self) -> Option<&str> {
        self.description()
    }

    async fn invoke_any(&self, tool_call: ToolCall) -> ToolMessage {
        let typed_input: T::Input = match serde_json::from_str(&tool_call.function.arguments) {
            Ok(input) => input,
            Err(e) => {
                return ToolMessage::builder()
                    .tool_call_id(&tool_call.id)
                    .content(ToolError::InputDeserializationFailed(e.to_string()).to_string())
                    .build();
            }
        };

        match self.invoke(typed_input).await {
            Ok(output) => match serde_json::to_value(output) {
                Ok(value) => ToolMessage::builder()
                    .tool_call_id(&tool_call.id)
                    .content(serde_json::to_string_pretty(&value).unwrap())
                    .build(),
                Err(e) => ToolMessage::builder()
                    .tool_call_id(&tool_call.id)
                    .content(ToolError::OutputSerializationFailed(e.to_string()).to_string())
                    .build(),
            },
            Err(e) => ToolMessage::builder()
                .tool_call_id(&tool_call.id)
                .content(e.to_string())
                .build(),
        }
    }

    fn input_schema(&self) -> Value {
        self.tool_schema()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde_json::Value;

    #[test]
    fn test_tool_schema() {
        #[derive(Default, Debug, Clone)]
        struct TestTool;

        #[derive(Debug, Clone, JsonSchema, Deserialize)]
        struct TestInput {
            a: String,
            b: f32,
        }

        #[async_trait]
        impl Tool for TestTool {
            type Input = TestInput;

            type Output = Value;

            type Error = String;

            fn name(&self) -> &str {
                "finish_tool"
            }

            async fn invoke(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
                todo!()
            }
        }
        let schema = TestTool.input_schema();
        dbg!(schema);
    }
}
