-- Items: the 172k Russian Construction Resources Classifier entries
CREATE TABLE IF NOT EXISTS items (
  id INTEGER PRIMARY KEY,
  sheet TEXT NOT NULL,
  code TEXT NOT NULL,
  name TEXT NOT NULL,
  unit TEXT,
  category TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_code ON items(code);
CREATE INDEX IF NOT EXISTS idx_items_sheet ON items(sheet);

-- FTS5 virtual table for fast text search with BM25 ranking and Russian
-- support via the unicode61 tokenizer + remove_diacritics for ё→е folding.
-- The "contentless-delete" form (content=items, content_rowid=id) keeps
-- the index in sync via INSERT/DELETE on items and saves space.
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
  name,
  content=items,
  content_rowid=id,
  tokenize="unicode61 remove_diacritics 1"
);

-- Triggers that keep the FTS index in sync with the items table.
CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
  INSERT INTO items_fts(rowid, name) VALUES (new.id, new.name);
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, name) VALUES('delete', old.id, old.name);
END;
CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, name) VALUES('delete', old.id, old.name);
  INSERT INTO items_fts(rowid, name) VALUES (new.id, new.name);
END;

-- Labelled examples: (query → KSR item) pairs. Grows from user clicks
-- (POST /api/feedback) and bulk ingest from xlsx.
CREATE TABLE IF NOT EXISTS examples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_text TEXT NOT NULL,
  query_norm TEXT NOT NULL,
  item_id INTEGER NOT NULL,
  source TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  added_at TEXT NOT NULL,
  FOREIGN KEY(item_id) REFERENCES items(id)
);
CREATE INDEX IF NOT EXISTS idx_examples_norm ON examples(query_norm);
CREATE INDEX IF NOT EXISTS idx_examples_item ON examples(item_id);

-- Same FTS index but over the example queries — used to find similar
-- past queries (and boost the items they were labelled with).
CREATE VIRTUAL TABLE IF NOT EXISTS examples_fts USING fts5(
  query_norm,
  content=examples,
  content_rowid=id,
  tokenize="unicode61 remove_diacritics 1"
);
CREATE TRIGGER IF NOT EXISTS examples_ai AFTER INSERT ON examples BEGIN
  INSERT INTO examples_fts(rowid, query_norm) VALUES (new.id, new.query_norm);
END;
CREATE TRIGGER IF NOT EXISTS examples_ad AFTER DELETE ON examples BEGIN
  INSERT INTO examples_fts(examples_fts, rowid, query_norm) VALUES('delete', old.id, old.query_norm);
END;
