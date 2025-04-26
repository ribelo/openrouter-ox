use parking_lot::{MappedMutexGuard, Mutex, MutexGuard};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::Arc,
};

use crate::tool::ToolBox;

#[derive(Clone, Default)]
pub struct AgentContext {
    pub deps: Arc<Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
    pub tools: Option<ToolBox>,
}

impl AgentContext {
    pub fn new() -> Self {
        AgentContext::default()
    }
    pub fn insert<T: Any + Send + Sync>(&self, value: T) -> Option<T> {
        let key = TypeId::of::<T>();
        self.deps
            .lock()
            .insert(key, Box::new(value))
            .and_then(|old_boxed_value| old_boxed_value.downcast::<T>().ok())
            .map(|boxed_t| *boxed_t)
    }
    pub fn get<T: Any + Send + Sync>(&self) -> Option<MappedMutexGuard<'_, T>> {
        let key = TypeId::of::<T>();
        let guard = self.deps.lock();
        MutexGuard::try_map(guard, |map| {
            // Get a mutable reference to the Box<dyn Any> if the key exists.
            map.get_mut(&key)
                // Then, try to downcast the mutable reference inside the Box to &mut T.
                .and_then(|boxed_value| boxed_value.downcast_mut::<T>())
        })
        .ok()
    }
}
