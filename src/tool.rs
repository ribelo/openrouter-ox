use std::{
    collections::HashMap,
    fmt::{self, Debug},
    future::Future,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use bon::Builder;
use schemars::{schema_for, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

use crate::{
    message::{Messages, ToolMessage},
    response::ToolCall,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDescription {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub name: String,
    pub parameters: Value,
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
pub struct ToolSchema {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadata {
    pub name: String,
    pub description: Option<String>,
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
    pub fn tool<T: Tool + 'static>(self, tool: T) -> Self {
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

    pub async fn invoke(&self, tool_call: &ToolCall) -> Messages {
        // Assuming name and id always exist as per the prompt.
        // Use as_deref() to borrow the content of the Option<String> as &str without moving.
        let tool_name = tool_call
            .function
            .name
            .as_deref()
            .expect("Tool function name is missing");
        let tool_call_id = tool_call.id.as_deref().expect("Tool call ID is missing");

        match self.get(tool_name) {
            Some(tool) => tool.invoke_any(tool_call).await,
            None => {
                // Use the borrowed tool_name and clone it to create the error message.
                // Use the borrowed tool_call_id and convert it to String for ToolMessage.
                ToolMessage {
                    content: ToolError::ToolNotFound(tool_name.to_string()).to_string(),
                    tool_call_id: tool_call_id.to_string(),
                }
                .into()
            }
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
    pub fn metadata(&self) -> Vec<ToolSchema> {
        self.tools
            .read()
            .unwrap()
            .values()
            .map(|tool| ToolSchema {
                tool_type: "function".to_string(),
                function: FunctionMetadata {
                    name: tool.name().to_string(),
                    description: tool.description().map(|s| s.to_string()),
                    parameters: tool.input_schema(),
                },
            })
            .collect()
    }

    pub fn filter_tools(&self) -> Vec<Arc<dyn AnyTool>> {
        self.tools
            .read()
            .unwrap()
            .values()
            .filter(|tool| !tool.is_subagent())
            .cloned()
            .collect()
    }

    pub fn filter_agents(&self) -> Vec<Arc<dyn AnyTool>> {
        self.tools
            .read()
            .unwrap()
            .values()
            .filter(|tool| tool.is_subagent())
            .cloned()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsList(Vec<ToolMetadata>);

#[async_trait]
pub trait AnyTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str>;
    async fn invoke_any(&self, tool_call: &ToolCall) -> Messages;
    fn input_schema(&self) -> Value;
    fn is_subagent(&self) -> bool;
}

#[async_trait]
pub trait Tool: Send + Sync {
    type Input: JsonSchema + DeserializeOwned + Send + Sync;
    type Error: ToString;
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str> {
        None
    }
    async fn invoke(&self, tool_call_id: &str, input: Self::Input)
        -> Result<Messages, Self::Error>;
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
    fn is_subagent(&self) -> bool {
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

    async fn invoke_any(&self, tool_call: &ToolCall) -> Messages {
        // Assumption: tool_call.id and tool_call.function.name are always Some.
        // Use expect() to enforce this assumption at runtime. Panics if violated.
        let tool_call_id_str = tool_call
            .id
            .as_deref()
            .expect("Tool call ID is missing, but was expected.");
        // let _tool_name = tool_call
        //     .function
        //     .name
        //     .as_deref()
        //     .expect("Tool function name is missing, but was expected."); // Name used implicitly by caller (ToolBox::invoke)

        let typed_input: T::Input = match serde_json::from_str(&tool_call.function.arguments) {
            Ok(input) => input,
            Err(e) => {
                // Deserialization failed, return ToolMessage with the correct (expected) ID
                return ToolMessage::new(
                    // Pass the ID as String
                    tool_call_id_str.to_string(),
                    ToolError::InputDeserializationFailed(e.to_string()).to_string(),
                )
                .into();
            }
        };

        // Invoke the actual tool implementation with &str tool_call_id
        match self.invoke(tool_call_id_str, typed_input).await {
            Ok(messages) => messages, // Success, return the result messages
            Err(e) => {
                // Invocation failed, return ToolMessage with the correct (expected) ID
                ToolMessage::new(
                    // Pass the ID as String
                    tool_call_id_str.to_string(),
                    e.to_string(), // Use the error from the tool
                )
                .into()
            }
        }
    }

    fn input_schema(&self) -> Value {
        self.tool_schema()
    }

    fn is_subagent(&self) -> bool {
        self.is_subagent()
    }
}

#[derive(Clone, Builder)]
pub struct SimpleTool<I, O, E, Fut>
where
    I: JsonSchema + Serialize + DeserializeOwned,
    O: Serialize,
    E: ToString,
    Fut: Future<Output = Result<O, E>> + Send + 'static,
{
    #[builder(into)]
    name: String,
    #[builder(into)]
    description: Option<String>,
    handler: fn(I) -> Fut,
}

// impl<I, O, E, Fut, S: simple_tool_builder::State> SimpleToolBuilder<I, O, E, Fut, S>
// where
//     I: JsonSchema + Serialize + DeserializeOwned,
//     O: Serialize,
//     E: ToString,
//     Fut: Future<Output = Result<O, E>> + Send + 'static,
// {
//     pub fn handler<F>(mut self, handler: F) -> Self
//     where
//         F: Fn(I) -> Fut + Send + Sync + 'static,
//     {
//         self.handler = Arc::new(handler);
//         self
//     }
// }

#[async_trait]
impl<I, O, E, Fut> Tool for SimpleTool<I, O, E, Fut>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    O: Serialize,
    E: ToString,
    Fut: Future<Output = Result<O, E>> + Send + 'static,
{
    type Input = I;
    type Error = String;

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    async fn invoke(
        &self,
        tool_call_id: &str,
        input: Self::Input,
    ) -> Result<Messages, Self::Error> {
        match (self.handler)(input).await {
            Ok(content) => {
                let serialized = serde_json::to_string(&content)
                    .map_err(|e| format!("Failed to serialize output: {}", e))?;

                Ok(ToolMessage::new(tool_call_id, serialized).into())
            }
            Err(err) => Err(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::message::UserMessage;

    use super::*;
    use async_trait::async_trait;
    use schemars::JsonSchema;

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
            type Error = String;

            fn name(&self) -> &str {
                "finish_tool"
            }

            async fn invoke(
                &self,
                tool_call_id: &str,
                input: Self::Input,
            ) -> Result<Messages, Self::Error> {
                Ok(UserMessage::new(vec!["finish".to_string()]).into())
            }
        }
        let schema = TestTool.input_schema();
        dbg!(schema);
    }
}
