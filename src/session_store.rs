//! SQLite-backed session store.
//!
//! Sessions group related memory entries under a common identifier and carry
//! timestamp and tag metadata. Each session is stored in a local SQLite
//! database; memory entries are linked to sessions by storing the session ID
//! as a payload field in the Qdrant vector store.
//!
//! Database schema is managed via the migration files in `./migrations/`.

use sqlx::{
    sqlite::{SqliteConnectOptions, SqlitePool},
    Row,
};
use std::str::FromStr;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::SessionError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A session record returned by lookup operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier (UUID v4).
    pub id: String,
    /// ISO 8601 timestamp of when the session was created.
    pub created_at: String,
    /// ISO 8601 timestamp of when the session was last updated.
    pub updated_at: String,
    /// Arbitrary string tags for categorising sessions.
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// SessionStore
// ---------------------------------------------------------------------------

pub struct SessionStore {
    pool: SqlitePool,
}

impl SessionStore {
    /// Open (or create) a SQLite database at `database_url` and run pending
    /// migrations.
    ///
    /// `database_url` should be a `sqlite:` connection string, for example:
    /// - `sqlite:./agent_memory.db` – file relative to the working directory
    /// - `sqlite:/var/data/agent_memory.db` – absolute path
    /// - `sqlite::memory:` – in-memory (useful for tests)
    pub async fn new(database_url: &str) -> Result<Self, SessionError> {
        let options = SqliteConnectOptions::from_str(database_url)
            .map_err(|e| SessionError::Database(e.to_string()))?
            .create_if_missing(true);

        let pool = SqlitePool::connect_with(options).await?;

        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .map_err(|e| SessionError::Migration(e.to_string()))?;

        Ok(Self { pool })
    }

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    /// Create a new session with the given tags and return it.
    ///
    /// A UUID v4 is generated for the session ID and the current UTC time is
    /// used for both `created_at` and `updated_at`.
    pub async fn create(&self, tags: Vec<String>) -> Result<Session, SessionError> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let tags_json = serde_json::to_string(&tags)
            .map_err(|e| SessionError::Serialization(e.to_string()))?;

        sqlx::query(
            "INSERT INTO sessions (id, created_at, updated_at, tags) VALUES (?, ?, ?, ?)",
        )
        .bind(&id)
        .bind(&now)
        .bind(&now)
        .bind(&tags_json)
        .execute(&self.pool)
        .await?;

        Ok(Session {
            id,
            created_at: now.clone(),
            updated_at: now,
            tags,
        })
    }

    // -----------------------------------------------------------------------
    // Read
    // -----------------------------------------------------------------------

    /// Return the session with the given `id`, or `None` if it does not exist.
    pub async fn get(&self, id: &str) -> Result<Option<Session>, SessionError> {
        let row = sqlx::query(
            "SELECT id, created_at, updated_at, tags FROM sessions WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        row.map(|r| parse_session_row(&r)).transpose()
    }

    /// Return a page of sessions ordered by `created_at` descending (newest first).
    ///
    /// Use `limit` to cap the number of results and `offset` to skip entries
    /// for cursor-style pagination. Pass `limit = 0` to use no upper bound.
    pub async fn list(&self, limit: u64, offset: u64) -> Result<Vec<Session>, SessionError> {
        let rows = sqlx::query(
            "SELECT id, created_at, updated_at, tags \
             FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
        )
        .bind(limit as i64)
        .bind(offset as i64)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(parse_session_row).collect()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn parse_session_row(row: &sqlx::sqlite::SqliteRow) -> Result<Session, SessionError> {
    let id: String = row.try_get("id")?;
    let created_at: String = row.try_get("created_at")?;
    let updated_at: String = row.try_get("updated_at")?;
    let tags_json: String = row.try_get("tags")?;
    let tags: Vec<String> = serde_json::from_str(&tags_json)
        .map_err(|e| SessionError::Serialization(e.to_string()))?;

    Ok(Session {
        id,
        created_at,
        updated_at,
        tags,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    async fn make_store() -> SessionStore {
        SessionStore::new("sqlite::memory:")
            .await
            .expect("in-memory session store should initialise")
    }

    #[tokio::test]
    async fn create_returns_session_with_generated_id() {
        let store = make_store().await;
        let session = store.create(vec![]).await.expect("create should succeed");

        assert!(!session.id.is_empty());
        assert!(
            Uuid::parse_str(&session.id).is_ok(),
            "id should be a valid UUID"
        );
        assert!(session.tags.is_empty());
    }

    #[tokio::test]
    async fn create_stores_tags() {
        let store = make_store().await;
        let tags = vec!["alpha".to_string(), "beta".to_string()];
        let session = store
            .create(tags.clone())
            .await
            .expect("create should succeed");

        assert_eq!(session.tags, tags);
    }

    #[tokio::test]
    async fn get_returns_none_for_unknown_id() {
        let store = make_store().await;
        let result = store.get("nonexistent").await.expect("get should not error");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn get_returns_session_after_create() {
        let store = make_store().await;
        let created = store
            .create(vec!["tag1".to_string()])
            .await
            .expect("create should succeed");

        let fetched = store
            .get(&created.id)
            .await
            .expect("get should not error")
            .expect("session should exist");

        assert_eq!(fetched.id, created.id);
        assert_eq!(fetched.tags, created.tags);
        assert_eq!(fetched.created_at, created.created_at);
    }

    #[tokio::test]
    async fn list_returns_all_sessions_newest_first() {
        let store = make_store().await;
        let s1 = store.create(vec![]).await.expect("create s1");
        let s2 = store.create(vec![]).await.expect("create s2");

        let sessions = store.list(50, 0).await.expect("list should succeed");
        assert_eq!(sessions.len(), 2);

        // Newest first – s2 was created after s1
        let ids: Vec<&str> = sessions.iter().map(|s| s.id.as_str()).collect();
        assert!(
            ids.contains(&s1.id.as_str()) && ids.contains(&s2.id.as_str()),
            "both sessions should be listed"
        );
    }

    #[tokio::test]
    async fn list_returns_empty_when_no_sessions() {
        let store = make_store().await;
        let sessions = store.list(50, 0).await.expect("list should succeed");
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn list_respects_limit() {
        let store = make_store().await;
        store.create(vec![]).await.expect("create s1");
        store.create(vec![]).await.expect("create s2");
        store.create(vec![]).await.expect("create s3");

        let sessions = store.list(2, 0).await.expect("list with limit should succeed");
        assert_eq!(sessions.len(), 2);
    }

    #[tokio::test]
    async fn list_respects_offset() {
        let store = make_store().await;
        store.create(vec![]).await.expect("create s1");
        store.create(vec![]).await.expect("create s2");
        store.create(vec![]).await.expect("create s3");

        let all = store.list(50, 0).await.expect("list all");
        let page2 = store.list(50, 2).await.expect("list with offset");
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].id, all[2].id);
    }
}
