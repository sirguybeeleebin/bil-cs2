CREATE EXTENSION IF NOT EXISTS pgcrypto;

INSERT INTO auth.services (client_id, client_secret_hash)
SELECT 'etl-service', crypt('supersecretpassword', gen_salt('bf'))
WHERE NOT EXISTS (
    SELECT 1 FROM auth.services WHERE client_id = 'etl-service'
);
