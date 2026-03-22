CREATE TABLE IF NOT EXISTS fashion_items (
    item_id INTEGER PRIMARY KEY,
    split TEXT NOT NULL,
    source_root_name TEXT NOT NULL,
    preprocessed_path TEXT NOT NULL,
    original_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    image_id TEXT NOT NULL,
    year_code TEXT NOT NULL,
    style TEXT NOT NULL,
    gender TEXT NOT NULL,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fashion_items_split ON fashion_items(split);
CREATE INDEX IF NOT EXISTS idx_fashion_items_label ON fashion_items(label);
CREATE INDEX IF NOT EXISTS idx_fashion_items_filename ON fashion_items(filename);
