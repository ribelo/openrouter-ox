use std::fmt;

use bon::{builder, Builder};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Builder)]
pub struct TextContent {
    pub text: String,
}

impl TextContent {
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl<T: Into<String>> From<T> for TextContent {
    fn from(text: T) -> Self {
        TextContent::new(text)
    }
}

impl fmt::Display for TextContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.text)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Builder, derive_more::Deref)]
pub struct ImageContent {
    #[deref]
    pub image_url: ImageUrl,
}

impl fmt::Display for ImageContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.image_url)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Builder, derive_more::Deref)]
pub struct ImageUrl {
    #[deref]
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl fmt::Display for ImageUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.url)
    }
}

impl ImageContent {
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }
}

pub trait ChatMessage: Serialize + DeserializeOwned {
    fn role(&self) -> Role;
    fn content(&self) -> impl IntoIterator<Item = &ContentPart>;
    fn content_mut(&mut self) -> impl IntoIterator<Item = &mut ContentPart>;
    fn push_content(&mut self, content: impl Into<ContentPart>);
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
}

#[derive(Debug, Serialize, Deserialize, derive_more::Display, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text(TextContent),
    Image(ImageContent),
    // ToolUse(ToolUse),
    // ToolResult(ToolResult),
}

impl<T: Into<String>> From<T> for ContentPart {
    fn from(s: T) -> Self {
        ContentPart::Text(TextContent { text: s.into() })
    }
}

#[derive(
    Debug,
    Default,
    Serialize,
    Deserialize,
    Clone,
    PartialEq,
    Eq,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::Into,
    derive_more::IntoIterator,
)]
pub struct Content(Vec<ContentPart>);

impl<T> FromIterator<T> for Content
where
    T: Into<ContentPart>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[serde(rename_all = "lowercase")]
pub struct SystemMessage {
    role: Role,
    #[builder(into)]
    content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl ChatMessage for SystemMessage {
    fn role(&self) -> Role {
        self.role
    }

    fn content(&self) -> impl IntoIterator<Item = &ContentPart> {
        self.content.0.iter()
    }

    fn content_mut(&mut self) -> impl IntoIterator<Item = &mut ContentPart> {
        self.content.0.iter_mut()
    }

    fn push_content(&mut self, content: impl Into<ContentPart>) {
        self.content.push(content.into());
    }

    fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    fn len(&self) -> usize {
        self.content.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct UserMessage {
    #[builder(into)]
    content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl ChatMessage for UserMessage {
    fn role(&self) -> Role {
        Role::User
    }

    fn content(&self) -> impl IntoIterator<Item = &ContentPart> {
        self.content.0.iter()
    }

    fn content_mut(&mut self) -> impl IntoIterator<Item = &mut ContentPart> {
        self.content.0.iter_mut()
    }

    fn push_content(&mut self, content: impl Into<ContentPart>) {
        self.content.push(content.into());
    }

    fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    fn len(&self) -> usize {
        self.content.len()
    }
}

impl UserMessage {
    pub fn new<T, U>(content: T) -> Self
    where
        T: IntoIterator<Item = U>,
        U: Into<ContentPart>,
    {
        Self {
            content: content.into_iter().map(Into::into).collect(),
            name: None,
        }
    }
}

impl<T, U> From<T> for UserMessage
where
    T: IntoIterator<Item = U>,
    U: Into<ContentPart>,
{
    fn from(content: T) -> Self {
        Self {
            content: content.into_iter().map(Into::into).collect(),
            name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct AssistantMessage {
    #[serde(default)]
    #[builder(into)]
    pub content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl<T, U> From<T> for AssistantMessage
where
    T: IntoIterator<Item = U>,
    U: Into<ContentPart>,
{
    fn from(content: T) -> Self {
        Self {
            content: content.into_iter().map(Into::into).collect(),
            name: None,
            tool_calls: None,
            refusal: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct ToolMessage {
    #[builder(into)]
    pub content: Vec<ContentPart>,
    pub tool_call_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
}

impl ChatMessage for Message {
    fn role(&self) -> Role {
        match self {
            Message::System(_) => Role::System,
            Message::User(_) => Role::User,
            Message::Assistant(_) => Role::Assistant,
            Message::Tool(_) => Role::Tool,
        }
    }

    fn content(&self) -> impl IntoIterator<Item = &ContentPart> {
        match self {
            Message::System(msg) => &msg.content,
            Message::User(msg) => &msg.content,
            Message::Assistant(msg) => &msg.content,
            Message::Tool(msg) => &msg.content,
        }
    }

    fn content_mut(&mut self) -> impl IntoIterator<Item = &mut ContentPart> {
        match self {
            Message::System(msg) => &mut msg.content,
            Message::User(msg) => &mut msg.content,
            Message::Assistant(msg) => &mut msg.content,
            Message::Tool(msg) => &mut msg.content,
        }
    }

    fn push_content(&mut self, content: impl Into<ContentPart>) {
        match self {
            Message::System(msg) => msg.content.push(content.into()),
            Message::User(msg) => msg.content.push(content.into()),
            Message::Assistant(msg) => msg.content.push(content.into()),
            Message::Tool(msg) => msg.content.push(content.into()),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Message::System(msg) => msg.content.is_empty(),
            Message::User(msg) => msg.content.is_empty(),
            Message::Assistant(msg) => msg.content.is_empty(),
            Message::Tool(msg) => msg.content.is_empty(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Message::System(msg) => msg.content.len(),
            Message::User(msg) => msg.content.len(),
            Message::Assistant(msg) => msg.content.len(),
            Message::Tool(msg) => msg.content.len(),
        }
    }
}

impl From<SystemMessage> for Message {
    fn from(message: SystemMessage) -> Self {
        Message::System(message)
    }
}

impl From<UserMessage> for Message {
    fn from(message: UserMessage) -> Self {
        Message::User(message)
    }
}

impl From<AssistantMessage> for Message {
    fn from(message: AssistantMessage) -> Self {
        Message::Assistant(message)
    }
}

#[derive(
    Debug,
    Clone,
    Default,
    Serialize,
    Deserialize,
    derive_more::Deref,
    derive_more::DerefMut,
    derive_more::IntoIterator,
    derive_more::Into,
)]
pub struct Messages(pub Vec<Message>);

impl From<Message> for Messages {
    fn from(value: Message) -> Self {
        Messages(vec![value])
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::message::UserMessage;

    use super::{
        AssistantMessage, Content, ContentPart, ImageContent, Message, SystemMessage, ToolMessage,
    };

    #[test]
    fn test_assistant_message_deserialization() {
        let json = json!({
            "content": [{
                "type": "text",
                "text": "Hello John! How can I assist you today?"
            }],
            "refusal": null,
            "role": "assistant"
        });

        let msg: AssistantMessage = serde_json::from_value(json).unwrap();
        assert_eq!(msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &msg.content[0] {
            assert_eq!(text_content.text, "Hello John! How can I assist you today?");
            assert!(msg.refusal.is_none());
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_system_message_deserialization() {
        let json = json!({
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant"
            }],
            "role": "system"
        });

        let msg: SystemMessage = serde_json::from_value(json).unwrap();
        assert_eq!(msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &msg.content[0] {
            assert_eq!(text_content.text, "You are a helpful assistant");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_user_message_deserialization() {
        let json = json!({
            "content": [{
                "type": "text",
                "text": "What is the weather?"
            }],
            "role": "user"
        });

        let msg: UserMessage = serde_json::from_value(json).unwrap();
        assert_eq!(msg.content.len(), 1);
        if let ContentPart::Text(text_content) = &msg.content[0] {
            assert_eq!(text_content.text, "What is the weather?");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_tool_message_deserialization() {
        let json = json!({
            "content": [{
                "type": "text",
                "text": "The temperature is 72F"
            }],
            "role": "tool",
            "tool_call_id": "weather_123"
        });

        let msg: ToolMessage = serde_json::from_value(json).unwrap();
        if let ContentPart::Text(text_content) = &msg.content[0] {
            assert_eq!(text_content.text, "The temperature is 72F");
            assert_eq!(msg.tool_call_id, "weather_123");
        } else {
            panic!("Expected text content");
        }
    }

    #[test]
    fn test_message_deserialization() {
        let json = json!({
            "content": [{
                "type": "text",
                "text": "Hello John! How can I assist you today?"
            }],
            "refusal": null,
            "role": "assistant"
        });

        let msg: Message = serde_json::from_value(json).unwrap();
        match msg {
            Message::Assistant(assistant_msg) => {
                if let ContentPart::Text(text_content) = &assistant_msg.content[0] {
                    assert_eq!(text_content.text, "Hello John! How can I assist you today?");
                    assert!(assistant_msg.refusal.is_none());
                } else {
                    panic!("Expected text content");
                }
            }
            _ => panic!("Expected assistant message"),
        }
    }
}
