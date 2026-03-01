-- Add index on created_at to speed up the ORDER BY created_at DESC queries
-- used by the session listing endpoint.

CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions (created_at DESC);
