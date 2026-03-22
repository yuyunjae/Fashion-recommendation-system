CREATE TABLE IF NOT EXISTS fashion_items (
    item_id BIGINT PRIMARY KEY,
    split VARCHAR(16) NOT NULL,
    source_root_name VARCHAR(32) NOT NULL,
    preprocessed_path TEXT NOT NULL,
    original_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    image_id VARCHAR(64) NOT NULL,
    year_code VARCHAR(8) NOT NULL,
    style VARCHAR(128) NOT NULL,
    gender VARCHAR(16) NOT NULL,
    label VARCHAR(256) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fashion_items_split ON fashion_items(split);
CREATE INDEX IF NOT EXISTS idx_fashion_items_label ON fashion_items(label);
CREATE INDEX IF NOT EXISTS idx_fashion_items_filename ON fashion_items(filename);
