-- +up
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE ml_results (
    ml_result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    predictor_path VARCHAR(255) NOT NULL,
    metrics_path VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

-- +down
DROP TABLE IF EXISTS ml_results;
