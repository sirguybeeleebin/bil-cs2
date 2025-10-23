-- +up
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE players (
    player_uuid UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id INT UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

-- +down
DROP TABLE IF EXISTS players;
