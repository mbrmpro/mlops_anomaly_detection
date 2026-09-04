CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    image_path TEXT NOT NULL UNIQUE,
    category VARCHAR(50) NOT NULL,
    split VARCHAR(20) NOT NULL,
    defect_type VARCHAR(50) NOT NULL,
    is_anomaly BOOLEAN NOT NULL
);
