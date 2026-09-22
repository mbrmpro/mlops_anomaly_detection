CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,

    image_path TEXT NOT NULL UNIQUE,

    category VARCHAR(50) NOT NULL
        CHECK (category IN ('bottle', 'wood', 'pill')),

    split VARCHAR(10) NOT NULL
        CHECK (split IN ('train', 'test')),

    source_split VARCHAR(10) NOT NULL
        CHECK (source_split IN ('train', 'test')),

    batch_id SMALLINT,

    defect_type VARCHAR(50) NOT NULL,

    is_anomaly BOOLEAN NOT NULL,

    is_available BOOLEAN NOT NULL DEFAULT FALSE,

    released_at TIMESTAMP,

    CHECK (
        (split = 'train' AND batch_id BETWEEN 1 AND 3)
        OR
        (split = 'test' AND batch_id IS NULL)
    )
);


CREATE INDEX IF NOT EXISTS idx_images_category
ON images(category);


CREATE INDEX IF NOT EXISTS idx_images_split
ON images(category, split);


CREATE INDEX IF NOT EXISTS idx_images_batch
ON images(category, batch_id, is_available);
