-- Create the sessions table for agent session tracking.
-- Sessions group related memory entries and carry timestamp and tag metadata.

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT NOT NULL PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    tags       TEXT NOT NULL DEFAULT '[]'
);
