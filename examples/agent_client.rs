//! Example agent client demonstrating the full memory lifecycle.
//!
//! Connects to a running `penr-oz-agent-memory` server and shows how to:
//!   1. Check server health
//!   2. Store memories with metadata and session tags
//!   3. Retrieve memories via semantic search
//!   4. Simulate an agent that recalls context before responding
//!   5. Delete a memory entry
//!
//! # Usage
//!
//! Start the server first:
//! ```sh
//! cargo run
//! ```
//!
//! Then run this example (optionally pass a custom server URL):
//! ```sh
//! cargo run --example agent_client [SERVER_URL]
//! ```
//!
//! `SERVER_URL` defaults to `http://127.0.0.1:8080`.

use std::collections::HashMap;

use reqwest::Client;
use serde::{Deserialize, Serialize};

const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8080";

// ---------------------------------------------------------------------------
// Request / response types mirroring the server API
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct StoreMemoryRequest {
    text: String,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    metadata: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session: Option<String>,
}

#[derive(Deserialize, Debug)]
struct StoreMemoryResponse {
    id: String,
}

#[derive(Deserialize, Debug)]
struct SearchResult {
    #[allow(dead_code)]
    id: String,
    text: String,
    score: f32,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

#[derive(Deserialize, Debug)]
struct SearchMemoryResponse {
    results: Vec<SearchResult>,
}

#[derive(Deserialize, Debug)]
struct HealthResponse {
    status: String,
    default_provider: String,
    providers: Vec<String>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_SERVER_URL.to_string());

    println!("Agent Memory Client Demo");
    println!("========================");
    println!("Server: {base_url}");
    println!();

    let client = Client::new();

    // -----------------------------------------------------------------------
    // Step 1: Health check
    // -----------------------------------------------------------------------
    println!("--- Step 1: Health Check ---");
    let health: HealthResponse = client
        .get(format!("{base_url}/health"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    println!("  status          : {}", health.status);
    println!("  default provider: {}", health.default_provider);
    println!("  providers       : {:?}", health.providers);
    println!();

    // -----------------------------------------------------------------------
    // Step 2: Store memories
    //
    // Simulates an agent that has learned facts about a project and wants to
    // persist them so they can be recalled later via semantic search.
    // -----------------------------------------------------------------------
    let session = "demo-agent-session";

    println!("--- Step 2: Storing Memories (session={session}) ---");
    let memories: &[(&str, &str)] = &[
        (
            "The user prefers concise explanations over verbose ones.",
            "preference",
        ),
        (
            "The project uses Rust for the backend and React for the frontend.",
            "tech-stack",
        ),
        (
            "The deployment target is AWS us-east-1 region.",
            "infrastructure",
        ),
        (
            "Authentication is handled via JWT tokens with a 24-hour expiry.",
            "security",
        ),
        (
            "The team follows trunk-based development with feature flags.",
            "workflow",
        ),
    ];

    let mut stored_ids: Vec<String> = Vec::new();

    for (text, tag) in memories {
        let mut metadata = HashMap::new();
        metadata.insert("tag".to_string(), tag.to_string());

        let resp: StoreMemoryResponse = client
            .post(format!("{base_url}/memory"))
            .json(&StoreMemoryRequest {
                text: text.to_string(),
                metadata,
                session: Some(session.to_string()),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        println!("  stored [{tag:14}] {} (id={}…)", text, &resp.id[..8]);
        stored_ids.push(resp.id);
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 3: Semantic search
    //
    // Queries are embedded by the server and matched against stored vectors
    // using cosine similarity.  Results are ranked by relevance score.
    // -----------------------------------------------------------------------
    println!("--- Step 3: Semantic Search ---");
    let queries = [
        "What cloud infrastructure does the project use?",
        "How does the team handle authentication and security?",
        "What is the user's communication style preference?",
    ];

    for query in &queries {
        println!("  query: \"{query}\"");

        let resp: SearchMemoryResponse = client
            .get(format!("{base_url}/memory/search"))
            .query(&[
                ("q", query.to_string()),
                ("session", session.to_string()),
                ("limit", "3".to_string()),
            ])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        for result in &resp.results {
            println!(
                "    [{:.3}] ({}) {}",
                result.score,
                result.metadata.get("tag").map(String::as_str).unwrap_or("-"),
                result.text,
            );
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Step 4: Simulated agent workflow
    //
    // The agent receives a task, recalls relevant memories for context, then
    // stores its synthesised response as a new memory entry.
    // -----------------------------------------------------------------------
    println!("--- Step 4: Simulated Agent Workflow ---");
    println!("  Task: 'Explain the deployment process'");

    let context_resp: SearchMemoryResponse = client
        .get(format!("{base_url}/memory/search"))
        .query(&[
            ("q", "deployment and infrastructure".to_string()),
            ("session", session.to_string()),
            ("limit", "2".to_string()),
        ])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    println!(
        "  Agent recalled {} relevant memories:",
        context_resp.results.len()
    );
    for r in &context_resp.results {
        println!("    - {}", r.text);
    }
    println!();

    let agent_response = "Deployment targets AWS us-east-1. \
        The team ships via trunk-based development with feature flags \
        to ensure safe, incremental rollouts.";

    println!("  Agent stores its synthesised response…");
    let new_mem: StoreMemoryResponse = client
        .post(format!("{base_url}/memory"))
        .json(&StoreMemoryRequest {
            text: agent_response.to_string(),
            metadata: {
                let mut m = HashMap::new();
                m.insert("tag".to_string(), "agent-response".to_string());
                m
            },
            session: Some(session.to_string()),
        })
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    println!("  Stored agent response (id={}…)", &new_mem.id[..8]);
    println!();

    // -----------------------------------------------------------------------
    // Step 5: Memory lifecycle – delete an entry
    // -----------------------------------------------------------------------
    println!("--- Step 5: Memory Lifecycle – Deleting a Memory ---");
    let id_to_delete = &stored_ids[0];
    client
        .delete(format!("{base_url}/memory/{id_to_delete}"))
        .send()
        .await?
        .error_for_status()?;
    println!("  Deleted memory id={}…", &id_to_delete[..8]);
    println!();

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("Demo complete.  Memory lifecycle demonstrated:");
    println!("  [ok] Health check");
    println!("  [ok] Store memories with metadata and session tags");
    println!("  [ok] Semantic search with session filtering");
    println!("  [ok] Agent context retrieval and response storage");
    println!("  [ok] Delete a memory entry");

    Ok(())
}
