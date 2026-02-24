use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::embedding::Embedding;

/// A single memory entry stored in the vector store.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    #[serde(skip)]
    pub embedding: Embedding,
}

/// A search result returned from a semantic similarity query.
#[derive(Clone, Debug, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    pub score: f32,
}

/// Thread-safe in-memory vector store for agent memory.
pub struct MemoryStore {
    entries: RwLock<HashMap<String, MemoryEntry>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Store a new memory entry, returning its generated ID.
    pub fn store(
        &self,
        text: String,
        metadata: HashMap<String, String>,
        session: Option<String>,
        embedding: Embedding,
    ) -> String {
        let id = Uuid::new_v4().to_string();
        let entry = MemoryEntry {
            id: id.clone(),
            text,
            metadata,
            session,
            embedding,
        };
        self.entries.write().unwrap().insert(id.clone(), entry);
        id
    }

    /// Search for the top-k most similar entries to the given query embedding.
    ///
    /// Results are ranked by cosine similarity in descending order.
    /// An optional `session` filter limits results to a specific session tag.
    pub fn search(
        &self,
        query_embedding: &Embedding,
        limit: usize,
        session: Option<&str>,
    ) -> Vec<SearchResult> {
        let entries = self.entries.read().unwrap();
        let mut scored: Vec<SearchResult> = entries
            .values()
            .filter(|e| match session {
                Some(s) => e.session.as_deref() == Some(s),
                None => true,
            })
            .map(|e| SearchResult {
                id: e.id.clone(),
                text: e.text.clone(),
                metadata: e.metadata.clone(),
                session: e.session.clone(),
                score: cosine_similarity(query_embedding, &e.embedding),
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }

    /// Delete a memory entry by ID. Returns `true` if the entry existed.
    pub fn delete(&self, id: &str) -> bool {
        self.entries.write().unwrap().remove(id).is_some()
    }
}

/// Compute the cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_search_returns_ranked_results() {
        let store = MemoryStore::new();
        store.store(
            "hello world".to_string(),
            HashMap::new(),
            None,
            vec![1.0, 0.0, 0.0],
        );
        store.store(
            "goodbye world".to_string(),
            HashMap::new(),
            None,
            vec![0.0, 1.0, 0.0],
        );
        store.store(
            "hello again".to_string(),
            HashMap::new(),
            None,
            vec![0.9, 0.1, 0.0],
        );

        let results = store.search(&vec![1.0, 0.0, 0.0], 10, None);
        assert_eq!(results.len(), 3);
        // "hello world" should be the top result (exact match)
        assert_eq!(results[0].text, "hello world");
        assert!((results[0].score - 1.0).abs() < 1e-6);
        // "hello again" should be second
        assert_eq!(results[1].text, "hello again");
        // "goodbye world" should be last (orthogonal)
        assert_eq!(results[2].text, "goodbye world");
        assert!((results[2].score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn search_respects_limit() {
        let store = MemoryStore::new();
        for i in 0..5 {
            store.store(
                format!("entry {i}"),
                HashMap::new(),
                None,
                vec![i as f32, 0.0],
            );
        }

        let results = store.search(&vec![1.0, 0.0], 2, None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_filters_by_session() {
        let store = MemoryStore::new();
        store.store(
            "session a".to_string(),
            HashMap::new(),
            Some("a".to_string()),
            vec![1.0, 0.0],
        );
        store.store(
            "session b".to_string(),
            HashMap::new(),
            Some("b".to_string()),
            vec![1.0, 0.0],
        );
        store.store(
            "no session".to_string(),
            HashMap::new(),
            None,
            vec![1.0, 0.0],
        );

        let results = store.search(&vec![1.0, 0.0], 10, Some("a"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "session a");
    }

    #[test]
    fn delete_removes_entry() {
        let store = MemoryStore::new();
        let id = store.store(
            "temp".to_string(),
            HashMap::new(),
            None,
            vec![1.0],
        );

        assert!(store.delete(&id));
        assert!(!store.delete(&id)); // second delete returns false

        let results = store.search(&vec![1.0], 10, None);
        assert!(results.is_empty());
    }

    #[test]
    fn cosine_similarity_of_identical_vectors_is_one() {
        let a = vec![1.0, 2.0, 3.0];
        let score = cosine_similarity(&a, &a);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_of_orthogonal_vectors_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_with_zero_vector_is_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn store_preserves_metadata() {
        let store = MemoryStore::new();
        let mut meta = HashMap::new();
        meta.insert("key".to_string(), "value".to_string());

        store.store("with meta".to_string(), meta, None, vec![1.0]);

        let results = store.search(&vec![1.0], 10, None);
        assert_eq!(results[0].metadata.get("key").unwrap(), "value");
    }
}
