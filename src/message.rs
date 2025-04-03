use std::fmt;

use bon::{builder, Builder};
use derive_more::Deref;
use serde::{Deserialize, Serialize};

use crate::response::ToolCall;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Builder, Deref)]
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

#[derive(Debug, Serialize, Deserialize, derive_more::Display, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text(TextContent),
    Image(ImageContent),
}

impl ContentPart {
    pub fn as_text(&self) -> Option<&TextContent> {
        match self {
            ContentPart::Text(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&ImageContent> {
        match self {
            ContentPart::Image(image) => Some(image),
            _ => None,
        }
    }
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
pub struct Content(pub Vec<ContentPart>);

impl<T> FromIterator<T> for Content
where
    T: Into<ContentPart>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

impl From<String> for Content {
    fn from(content: String) -> Self {
        Self(vec![ContentPart::Text(TextContent { text: content })])
    }
}

impl From<&str> for Content {
    fn from(content: &str) -> Self {
        Self(vec![ContentPart::Text(TextContent {
            text: content.to_string(),
        })])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[serde(rename_all = "lowercase")]
pub struct SystemMessage {
    #[builder(into)]
    content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl SystemMessage {
    pub fn content(&self) -> &Content {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut Content {
        &mut self.content
    }

    pub fn push_content(&mut self, content: impl Into<ContentPart>) {
        self.content.push(content.into());
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }
}

impl SystemMessage {
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

impl<T, U> From<T> for SystemMessage
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
pub struct UserMessage {
    #[builder(field)]
    content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

impl<S: user_message_builder::State> UserMessageBuilder<S> {
    pub fn part(mut self, p: impl Into<ContentPart>) -> Self {
        self.content.push(p.into());
        self
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

    pub fn content(&self) -> &Content {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut Content {
        &mut self.content
    }

    pub fn push_content(&mut self, content: impl Into<ContentPart>) {
        self.content.push(content.into());
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn len(&self) -> usize {
        self.content.len()
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl AssistantMessage {
    pub fn new<T, U>(content: T) -> Self
    where
        T: IntoIterator<Item = U>,
        U: Into<ContentPart>,
    {
        Self {
            content: content.into_iter().map(Into::into).collect(),
            name: None,
            tool_calls: None,
            refusal: None,
        }
    }

    pub fn content(&self) -> &Content {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut Content {
        &mut self.content
    }

    pub fn push_content(&mut self, content: impl Into<ContentPart>) {
        self.content.push(content.into());
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }
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
    #[builder(field)]
    pub content: String,
    #[builder(into)]
    pub tool_call_id: String,
}

impl<S: tool_message_builder::State> ToolMessageBuilder<S> {
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }
}

impl ToolMessage {
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        }
    }

    pub fn content(&self) -> &String {
        &self.content
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }
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

impl Message {
    pub fn system(content: impl Into<Content>) -> Self {
        Message::System(SystemMessage::new(content.into()))
    }

    pub fn user(content: impl Into<Content>) -> Self {
        Message::User(UserMessage::new(content.into()))
    }

    pub fn assistant(content: impl Into<Content>) -> Self {
        Message::Assistant(AssistantMessage::new(content.into()))
    }

    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        let assistant_msg = match content {
            Some(content_text) if !content_text.is_empty() => {
                AssistantMessage {
                    content: Content::from(content_text),
                    name: None,
                    tool_calls,
                    refusal: None,
                }
            },
            _ => {
                AssistantMessage {
                    content: Content(Vec::new()),
                    name: None,
                    tool_calls,
                    refusal: None,
                }
            }
        };

        Message::Assistant(assistant_msg)
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

impl From<ToolMessage> for Message {
    fn from(message: ToolMessage) -> Self {
        Message::Tool(message)
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

impl Messages {
    pub fn push(&mut self, message: impl Into<Message>) {
        self.0.push(message.into());
    }

    pub fn insert(&mut self, index: usize, message: impl Into<Message>) {
        self.0.insert(index, message.into());
    }
}

impl From<Message> for Messages {
    fn from(value: Message) -> Self {
        Messages(vec![value])
    }
}

impl From<UserMessage> for Messages {
    fn from(value: UserMessage) -> Self {
        Messages(vec![Message::User(value)])
    }
}

impl From<AssistantMessage> for Messages {
    fn from(value: AssistantMessage) -> Self {
        Messages(vec![Message::Assistant(value)])
    }
}

impl From<SystemMessage> for Messages {
    fn from(value: SystemMessage) -> Self {
        Messages(vec![Message::System(value)])
    }
}

impl From<ToolMessage> for Messages {
    fn from(value: ToolMessage) -> Self {
        Messages(vec![Message::Tool(value)])
    }
}

impl FromIterator<Message> for Messages {
    fn from_iter<I: IntoIterator<Item = Message>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
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
            "content": "The temperature is 72F",
            "role": "tool",
            "tool_call_id": "weather_123"
        });

        let msg: ToolMessage = serde_json::from_value(json).unwrap();
        assert_eq!(msg.content, "The temperature is 72F");
        assert_eq!(msg.tool_call_id, "weather_123");
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
