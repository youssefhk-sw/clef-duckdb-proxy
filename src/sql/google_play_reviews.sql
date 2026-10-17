-- 1. Connect DuckDB to the local proxy ---------------------------------------
INSTALL jev FROM community;
LOAD jev;
-- Endpoint where jev extension send request (clef proxy)
SET jev_api_url = 'http://127.0.0.1:8787/v1/systemone';
SET jev_api_key = 'local';                   
SET jev_max_rows_per_statement = 1500;   

-- 2. Build Table Samples ---------------------------------------
CREATE OR REPLACE TABLE apps_reviews AS
SELECT
    ROW_NUMBER() OVER (ORDER BY app_id, user_name) AS id,
    app_id,
    CASE
        WHEN LENGTH(user_name) <= 3 THEN user_name
        ELSE
            LEFT(user_name, 2)
            || REPEAT('*', LENGTH(user_name) - 3)
            || RIGHT(user_name, 1)
    END AS user_name,
    content
FROM (
    SELECT
        appId AS app_id,
        userName AS user_name,
        content
    FROM read_csv('reviews.csv') -- apps reviews data from (https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews)
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY appId
        ORDER BY userName
    ) <= 100
);

-- 3. Classify with Clef (via the proxy) -------------------------------------
CREATE OR REPLACE TABLE predictions AS
SELECT ar.id,
       CAST(jev_choice(ar, 'What is the feelling of the user about the app?',
                       ['positive', 'negative', 'other']) AS VARCHAR) AS sentiment
FROM (SELECT id, content FROM apps_reviews) AS ar;

-- 4. The most satisfied app -------------------------------------
SELECT
    r.app_id,
    COUNT(*) AS positive_reviews
FROM apps_reviews AS r
JOIN predictions AS p
    ON r.id = p.id
WHERE p.sentiment = 'positive' -- the values already lowercase
GROUP BY r.app_id
ORDER BY positive_reviews DESC;

-- 5. App problems type -------------------------------------

SELECT
    app_id,
    COUNT(*) AS total_reviews,
    ROUND(AVG(satisfaction_score), 3) AS satisfaction_score
FROM (
    SELECT
        r.app_id,
        jev_score_norm(
            r,
            'How severe is the problem described in this review?',
            ['no problem', 'minor inconvenience', 'moderate problem', 'serious problem', 'critical problem']
        ) AS satisfaction_score
    FROM apps_reviews AS r
) AS scored
GROUP BY app_id
ORDER BY satisfaction_score DESC;