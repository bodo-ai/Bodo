WITH 
dollar_strings AS (
    SELECT L.index + 1 as idx1, L.value::varchar as strs1
    FROM table(flatten([
        $$string with a ' character$$,
        $$regular expression with \ characters: \d{2}-\d{3}-\d{4}$$,
        $$string with a newline
character$$,
        $$A$B$C$D$$
    ])) L
),
quote_strings AS (
    SELECT L.index + 1 as idx2, L.value::varchar as strs2
    FROM table(flatten([
        'string with a \' character',
        'regular expression with \\ characters: \\d{2}-\\d{3}-\\d{4}',
        'string with a newline\ncharacter',
        'A$B$C$D'
    ])) L
)
SELECT
    idx1, idx2, length(strs1) as L
FROM dollar_strings
INNER JOIN quote_strings
ON dollar_strings.strs1 = quote_strings.strs2
ORDER BY idx1, idx2
