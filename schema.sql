BEGIN;
CREATE TABLE IF NOT EXISTS chats (
 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
 source TEXT NOT NULL DEFAULT 'demo', external_chat_id TEXT, advertisement TEXT,
 status TEXT NOT NULL DEFAULT 'new' CHECK(status IN ('new','qualifying','application','needs_manager','not_target')),
 client_name TEXT, phone TEXT, company TEXT, purpose TEXT,
 parameters JSONB NOT NULL DEFAULT '{}'::jsonb, summary TEXT,
 consent_given_at TIMESTAMPTZ, manager_name TEXT, bot_enabled BOOLEAN NOT NULL DEFAULT TRUE,
 paused_until TIMESTAMPTZ, telegram_message_id BIGINT,
 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
 UNIQUE(source,external_chat_id)
);
CREATE TABLE IF NOT EXISTS messages (
 id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 chat_id UUID NOT NULL REFERENCES chats(id),
 role TEXT NOT NULL CHECK(role IN ('user','assistant','manager')),
 content TEXT NOT NULL CHECK(LENGTH(TRIM(content))>0),
 external_message_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
 UNIQUE(chat_id,external_message_id)
);
CREATE INDEX IF NOT EXISTS messages_chat_id_id_idx ON messages(chat_id,id);
COMMIT;
