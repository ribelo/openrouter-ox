use bon::Builder;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_stream::Stream;

use crate::{
    message::{Message, Messages},
    provider_preference::ProviderPreferences,
    response::{ChatCompletionChunk, ChatCompletionResponse},
    tool::{ToolBox, ToolChoice},
    ApiRequestError, ErrorResponse, OpenRouter, BASE_URL,
};

const API_URL: &str = "api/v1/chat/completions";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub r#type: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Builder)]
pub struct Request {
    #[builder(field)]
    pub messages: Messages,
    #[builder(field)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[builder(into)]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(into)]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolBox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_a: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Prediction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transforms: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderPreferences>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    #[serde(skip)]
    pub open_router: OpenRouter,
}

impl<S: request_builder::State> RequestBuilder<S> {
    pub fn messages(mut self, messages: impl IntoIterator<Item = impl Into<Message>>) -> Self {
        self.messages = messages.into_iter().map(Into::into).collect();
        self
    }
    pub fn message(mut self, message: impl Into<Message>) -> Self {
        self.messages.push(message.into());
        self
    }
    pub fn response_format<T: JsonSchema + Serialize>(mut self) -> Self {
        let type_name = std::any::type_name::<T>().split("::").last().unwrap();
        let json_schema = schema_for!(T);
        let response_format = json!({
            "type": "json_schema",
            "json_schema": {"name": type_name, "schema": json_schema},
        });
        self.response_format = Some(response_format);
        self
    }
}

impl Request {
    pub fn push_message(&mut self, message: impl Into<Message>) {
        self.messages.push(message.into());
    }
    pub async fn send(&self) -> Result<ChatCompletionResponse, ApiRequestError> {
        let body = serde_json::to_value(self).unwrap();
        println!("{}", serde_json::to_string_pretty(&body).unwrap());
        let url = format!("{}/{}", BASE_URL, API_URL);
        let req = self
            .open_router
            .client
            .post(&url)
            .header(
                "Authorization",
                format!("Bearer {}", self.open_router.api_key),
            )
            .header("Content-Type", "application/json")
            .json(self);
        let res = req.send().await.unwrap();
        if res.status().is_success() {
            let text = res.text().await.unwrap();
            dbg!(&text);
            let data: crate::response::ChatCompletionResponse =
                serde_json::from_str(&text).unwrap();
            Ok(data)
        } else {
            let error_response: ErrorResponse = res.json().await?;
            Err(ApiRequestError::InvalidRequestError(error_response))
        }
    }

    pub async fn stream(
        &self,
    ) -> impl tokio_stream::Stream<Item = Result<ChatCompletionChunk, ApiRequestError>> {
        use tokio_stream::StreamExt;
        let url = format!("{}/{}", BASE_URL, API_URL);
        let mut body = serde_json::to_value(self).unwrap();
        body["stream"] = serde_json::Value::Bool(true);

        let response = self
            .open_router
            .client
            .post(&url)
            .bearer_auth(&self.open_router.api_key)
            .json(&body)
            .send()
            .await
            .unwrap();
        let byte_stream = response.bytes_stream();

        let parsed_stream = byte_stream.filter_map(move |chunk_result| {
            let chunk = match chunk_result {
                Ok(bytes) => bytes,
                Err(e) => return Some(Err(ApiRequestError::Stream(e.to_string()))),
            };
            let chunk_str = match String::from_utf8(chunk.to_vec()) {
                Ok(s) => s,
                Err(e) => return Some(Err(ApiRequestError::Stream(e.to_string()))),
            };
            ChatCompletionChunk::from_streaming_line(&chunk_str)
        });

        Box::pin(parsed_stream)
    }
}

// impl TokenCount for Message {
//     fn token_count(&self) -> usize {
//         match self {
//             Message::System(message) => message.content.token_count(),
//             Message::User(message) => message.content.token_count(),
//             Message::Assistant(message) => message.content.token_count(),
//             Message::Tool(message) => message.content.token_count(),
//         }
//     }
// }

// impl TokenCount for Messages {
//     fn token_count(&self) -> usize {
//         self.0.iter().map(|m| m.token_count()).sum()
//     }
// }

impl OpenRouter {
    pub fn chat_completion(&self) -> RequestBuilder<request_builder::SetOpenRouter> {
        Request::builder().open_router(self.clone())
    }
}

#[cfg(test)]
mod test {

    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use tokio_stream::StreamExt;

    use crate::{
        message::{Messages, UserMessage},
        tool::{Tool, ToolBox},
        OpenRouter,
    };

    #[tokio::test]
    async fn test_plain_request() {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let client = reqwest::Client::new();

        let payload = serde_json::json!({
            "model": "anthropic/claude-3.5-sonnet:beta",
            "temperature": 1,
            "messages": [{
                "role": "user",
                "content": "Hello!"
            }]
        });
        dbg!(&payload);

        let response = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .unwrap();

        let json = response.json::<serde_json::Value>().await.unwrap();
    }

    #[tokio::test]
    async fn test_chat_no_stream() {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let openrouter = OpenRouter::builder()
            .api_key(api_key)
            .client(client)
            .build();
        let res = openrouter
            .chat_completion()
            .model("google/gemini-2.0-flash-001")
            .message(UserMessage::from(vec!["Hi, I'm John."]))
            .build()
            .send()
            .await;
        dbg!(&res);
    }

    #[tokio::test]
    async fn test_chat_stream() {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let openai = OpenRouter::builder()
            .api_key(api_key)
            .client(client)
            .build();
        let mut res = openai
            .chat_completion()
            .model("anthropic/claude-3.5-sonnet:beta")
            .message(UserMessage::from(vec!["Hi, I'm John."]))
            .build()
            .stream()
            .await;
        while let Some(res) = res.next().await {
            dbg!(res);
        }
    }

    #[tokio::test]
    async fn test_response_format() {
        #[derive(Debug, Clone, Serialize, JsonSchema)]
        struct MyResult {
            expresion: String,
            result: String,
        }
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let openrouter = OpenRouter::builder()
            .api_key(api_key)
            .client(client)
            .build();
        let res = openrouter
            .chat_completion()
            .model("openai/gpt-4o-2024-11-20")
            .message(UserMessage::from(vec!["ile to jest 2+2"]))
            .response_format::<MyResult>()
            .build()
            .send()
            .await;
        dbg!(&res);
    }

    #[tokio::test]
    async fn test_openai() {
        let payload = json!({
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": "This is a test, use adding tool to verify if everything is working."
                }
              ]
            },
            {
              "role": "assistant",
              "content": "",
              "tool_calls": [
                {
                  "id": "call_sJgwcXOX1wXbylui3t53YFU6",
                  "type": "function",
                  "function": {
                    "name": "adding_tool",
                    "arguments": "{\"a\":5,\"b\":7}"
                  }
                }
              ]
            },
            {
              "role": "tool",
              "content": "{\"result\":12.0}",
              "tool_call_id": "call_sJgwcXOX1wXbylui3t53YFU6"
            }
          ],
          "model": "openai/gpt-4o-2024-11-20",
          "tools": [
            {
              "type": "function",
              "function": {
                "name": "adding_tool",
                "parameters": {
                  "$schema": "https://json-schema.org/draft/2020-12/schema",
                  "properties": {
                    "a": {
                      "format": "float",
                      "type": "number"
                    },
                    "b": {
                      "format": "float",
                      "type": "number"
                    }
                  },
                  "required": [
                    "a",
                    "b"
                  ],
                  "type": "object"
                }
              }
            }
          ]
        });

        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let client = reqwest::Client::new();

        let response = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&payload)
            .send()
            .await
            .unwrap();

        let json = response.json::<serde_json::Value>().await.unwrap();
        dbg!(json);
    }

    #[tokio::test]
    async fn test_adding_tool() {
        #[derive(Debug, Clone)]
        pub struct AddingTool;

        #[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
        pub struct AddingToolInput {
            a: f32,
            b: f32,
        }

        #[async_trait::async_trait]
        impl Tool for AddingTool {
            type Input = AddingToolInput;
            type Error = String;

            fn name(&self) -> &str {
                "adding_tool"
            }

            async fn invoke(
                &self,
                tool_call_id: &str,
                input: Self::Input,
            ) -> Result<Messages, Self::Error> {
                Ok(UserMessage::new(vec![json!({"result": input.a + input.b}).to_string()]).into())
            }
        }

        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let openrouter = OpenRouter::builder()
            .api_key(api_key)
            .client(client)
            .build();
        let tools = ToolBox::builder().tool(AddingTool).build();
        let mut messages = Messages::from(UserMessage::from(vec![
            "This is a test, use adding tool to verify if everything is working.",
        ]));
        let res = openrouter
            .chat_completion()
            .model("openai/gpt-4o-2024-11-20")
            .tools(tools.clone())
            .messages(messages.clone())
            .build()
            .send()
            .await
            .unwrap();
        messages.push(res.clone());
        if let Some(tool_calls) = res.tool_calls() {
            for tool_call in tool_calls {
                let msg = tools.invoke(&tool_call).await;
                messages.extend(msg);
            }
        }
        let res = openrouter
            .chat_completion()
            .model("openai/gpt-4o-2024-11-20")
            .tools(tools.clone())
            .messages(messages)
            .build()
            .send()
            .await
            .unwrap();
        dbg!(&res);
    }

    #[tokio::test]
    async fn test_wikipedia_tool() {
        #[derive(Debug, Clone)]
        pub struct Wikipedia;

        #[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
        pub struct WikipediaInput {
            query: String,
        }

        #[async_trait::async_trait]
        impl Tool for Wikipedia {
            type Input = WikipediaInput;
            type Error = String;

            fn name(&self) -> &str {
                "wikipedia"
            }

            async fn invoke(
                &self,
                tool_call_id: &str,
                input: Self::Input,
            ) -> Result<Messages, Self::Error> {
                let url = format!(
                            "https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={}",
                            input.query
                        );
                let res = reqwest::get(&url)
                    .await
                    .map_err(|e| e.to_string())?
                    .json::<serde_json::Value>()
                    .await
                    .map_err(|e| e.to_string())?;
                let msgs = UserMessage::new(vec![res.to_string()]);
                Ok(msgs.into())
            }
        }

        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        let client = reqwest::Client::new();
        let openrouter = OpenRouter::builder()
            .api_key(api_key)
            .client(client)
            .build();
        let tools = ToolBox::builder().tool(Wikipedia).build();
        let mut messages = Messages::from(UserMessage::from(vec![
            "Search Apollo project on Wikipedia.",
        ]));
        let res = openrouter
            .chat_completion()
            .model("openai/gpt-4o-2024-11-20")
            .tools(tools.clone())
            .messages(messages.clone())
            .build()
            .send()
            .await
            .unwrap();
        messages.push(res.clone());
        if let Some(tool_calls) = res.tool_calls() {
            for tool_call in tool_calls {
                let msg = tools.invoke(&tool_call).await;
                messages.extend(msg);
            }
        }
        let res = openrouter
            .chat_completion()
            .model("openai/gpt-4o-2024-11-20")
            .tools(tools.clone())
            .messages(messages)
            .build()
            .send()
            .await
            .unwrap();
        dbg!(&res);
    }

    #[tokio::test]
    async fn test_json_output() {
        #[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
        pub struct TestOutput {
            probability: f64,
            explanation: String,
        }

        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap();
        let openrouter = OpenRouter::builder().api_key(api_key).build();
        let mut messages = Messages::from(UserMessage::from(vec![
            "Can we live forever in the future?",
        ]));
        let res = openrouter
            .chat_completion()
            .model("google/gemini-2.0-flash-001")
            .messages(messages.clone())
            .response_format::<TestOutput>()
            .build()
            .send()
            .await
            .unwrap();
        dbg!(&res);
    }
}
