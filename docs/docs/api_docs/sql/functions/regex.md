# Regex Functions

BodoSQL currently uses Python's regular expression library via the `re`
module. Although this may be subject to change, it means that there are
several deviations from the behavior of Snowflake's regular expression
functions [(see here for snowflake documentation)](https://docs.snowflake.com/en/sql-reference/functions-regexp.html).
The key points and major deviations are noted below:

- Snowflake uses a superset of the POSIX ERE regular expression syntax. This means that BodoSQL can utilize several syntactic forms of regular expressions that Snowflake cannot [(see here for Python re documentation)](https://docs.python.org/3/library/re.html). However, there are several features that POSIX ERE has that Python's `re` does not:

  - POSIX character classes [(see here for a full list)](https://en.wikipedia.org/wiki/Regular_expression#Character_classes). BodoSQL does support these as macros for character sets. In other words, `[[:lower:]]` is transformed into `[a-z]`. However, this form of replacement cannot be escaped. Additionally, any character classes that are supposed to include the null terminator `\x00` instead start at `\x01`

  - Equivalence classes (not supported by BodoSQL).

  - Returning the longest match when using alternation patterns (BodoSQL returns the leftmost match).

- The regex functions can optionally take in a flag argument. The flag is a string whose characters control how matches to patterns occur. The following characters have meaning when contained in the flag string:

  - `'c'`: case-sensitive matching (the default behavior)
  - `'i'`: case-insensitive matching (if both 'c' and 'i' are provided, whichever one occurs last is used)
  - `'m'`: allows anchor patterns to interact with the start/end of each line, not just the start/end of the entire string.
  - `'s'`: allows the `.` metacharacter to capture newline characters
  - `'e'`: see `REGEXP_SUBSTR`/`REGEXP_INSTR`

- Currently, BodoSQL supports the lazy `?` operator whereas Snowflake does not. So for example, in Snowflake, the pattern \`\`(.\*?),'\` would match with as many characters as possible so long as the last character was a comma. However, in BodoSQL, the match would end as soon as the first comma.

- Currently, BodoSQL supports the following regexp features which should crash when done in Snowflake: `(?...)`, `\A`, `\Z`, `\1`, `\2`, `\3`, etc.

- Currently, BodoSQL requires the pattern argument and the flag argument (if provided) to be string literals as opposed to columns or expressions.

- Currently, extra backslashes may be required to escape certain characters if they have meaning in Python. The amount of backslashes required to properly escape a character depends on the usage.

- All matches are non-overlapping.

- If any of the numeric arguments are zero or negative, or the `group_num` argument is out of bounds, an error is raised. The only exception is `#!sql REGEXP_REPLACE`, which allows its occurrence argument to be zero.

BodoSQL currently supports the following regex functions:

#### REGEXP_LIKE

- `#!sql REGEXP_LIKE(str, pattern[, flag])`

  Returns `true` if the entire string matches with the pattern.
  If `flag` is not provided, `''` is used.

  If the pattern is empty, then `true` is returned if
  the string is also empty.

  For example:

  - 2 arguments: Returns `true` if `A` is a 5-character string where the first character is an a,
    the last character is a z, and the middle 3 characters are also lowercase characters (case-sensitive).

  ```sql
  SELECT REGEXP_LIKE(A, 'a[a-z]{3}z')
  ```

  - 3 arguments: Returns `true` if `A` starts with the letters `'THE'` (case-insensitive).

  ```sql
  SELECT REGEXP_LIKE(A, 'THE.*', 'i')
  ```

#### REGEXP_COUNT

- `#!sql REGEXP_COUNT(str, pattern[, position[, flag]])`

  Returns the number of times the string contains matches
  to the pattern, starting at the location specified
  by the `position` argument (with 1-indexing).
  If `position` is not provided, `1` is used.
  If `flag` is not provided, `''` is used.

  If the pattern is empty, 0 is returned.

  For example:

  - 2 arguments: Returns the number of times that any letters occur in `A`.

  ```sql
  SELECT REGEXP_COUNT(A, '[[:alpha:]]')
  ```

  - 3 arguments: Returns the number of times that any digit characters occur in `A`, not including
    the first 5 characters.

  ```sql
  SELECT REGEXP_COUNT(A, '\d', 6)
  ```

  - 4 arguments: Returns the number of times that a substring occurs in `A` that contains two
    ones with any character (including newlines) in between.

  ```sql
  SELECT REGEXP_COUNT(A, '1.1', 1, 's')
  ```

#### REGEXP_REPLACE

- `#!sql REGEXP_REPLACE(str, pattern[, replacement[, position[, occurrence[, flag]]]])`

  Returns the version of the inputted string where each
  match to the pattern is replaced by the replacement string,
  starting at the location specified by the `position` argument
  (with 1-indexing). The occurrence argument specifies which
  match to replace, where 0 means replace all occurrences. If
  `replacement` is not provided, `''` is used. If `position` is
  not provided, `1` is used. If `occurrence` is not provided,
  `0` is used. If `flag` is not provided, `''` is used.

  If there are an insufficient number of matches, or the pattern is empty,
  the original string is returned.

  !!! note
  back-references in the replacement pattern are supported,
  but may require additional backslashes to work correctly.

  For example:

  - 2 arguments: Deletes all whitespace in `A`.

  ```sql
  SELECT REGEXP_REPLACE(A, '[[:space:]]')
  ```

  - 3 arguments: Replaces all occurrences of `'hate'` in `A` with `'love'` (case-sensitive).

  ```sql
  SELECT REGEXP_REPLACE(A, 'hate', 'love')
  ```

  - 4 arguments: Replaces all occurrences of two consecutive digits in `A` with the same two
    digits reversed, excluding the first 2 characters.

  ```sql
  SELECT REGEXP_REPLACE(A, '(\d)(\d)', '\\\\2\\\\1', 3)
  ```

  - 5 arguments: Replaces the first character in `A` with an underscore.

  ```sql
  SELECT REGEXP_REPLACE(A, '^.', '_', 1, 2)
  ```

  - 6 arguments: Removes the first and last word from each line of `A` that contains
    at least 3 words.

  ```sql
  SELECT REGEXP_REPLACE(A, '^\w+ (.*) \w+$', '\\\\1', 0, 1, 'm')
  ```

#### REGEXP_SUBSTR

- `#!sql REGEXP_SUBSTR(str, pattern[, position[, occurrence[, flag[, group_num]]]])`

  Returns the substring of the original string that caused a
  match with the pattern, starting at the location specified
  by the `position` argument (with 1-indexing). The occurrence argument
  specifies which match to extract (with 1-indexing). If `position` is
  not provided, `1` is used. If `occurrence` is not provided,
  `1` is used. If `flag` is not provided, `''` is used. If `group_num`
  is not provided, and `flag` contains `'e`', `1` is used. If `group_num` is provided but the
  flag does not contain `e`, then it behaves as if it did. If the flag does contain `e`,
  then one of the subgroups of the match is returned instead of the entire match. The
  subgroup returned corresponds to the `group_num` argument
  (with 1-indexing).

  If there are an insufficient number of matches, or the pattern is empty,
  `NULL` is returned.

  For example:

  - 2 arguments: Returns the first number that occurs inside of `A`.

  ```sql
  SELECT REGEXP_SUBSTR(A, '\d+')
  ```

  - 3 arguments: Returns the first punctuation symbol that occurs inside of `A`, excluding the first 10 characters.

  ```sql
  SELECT REGEXP_SUBSTR(A, '[[:punct:]]', 11)
  ```

  - 4 arguments: Returns the fourth occurrence of two consecutive lowercase vowels in `A`.

  ```sql
  SELECT REGEXP_SUBSTR(A, '[aeiou]{2}', 1, 4)
  ```

  - 5 arguments: Returns the first 3+ character substring of `A` that starts with and ends with a vowel (case-insensitive, and
    it can contain newline characters).

  ```sql
  SELECT REGEXP_SUBSTR(A, '[aeiou].+[aeiou]', 1, 1, 'im')
  ```

  - 6 arguments: Looks for third occurrence in `A` of a number followed by a colon, a space, and a word
    that starts with `'a'` (case-sensitive) and returns the word that starts with `'a'`.

  ```sql
  SELECT REGEXP_SUBSTR(A, '(\d+): (a\w+)', 1, 3, 'e', 2)
  ```

#### REGEXP_INSTR

- `#!sql REGEXP_INSTR(str, pattern[, position[, occurrence[, option[, flag[, group_num]]]]])`

  Returns the location within the original string that caused a
  match with the pattern, starting at the location specified
  by the `position` argument (with 1-indexing). The occurrence argument
  specifies which match to extract (with 1-indexing). The option argument
  specifies whether to return the start of the match (if `0`) or the first
  location after the end of the match (if `1`). If `position` is
  not provided, `1` is used. If `occurrence` is not provided,
  `1` is used. If `option` is not provided, `0` is used. If `flag` is not
  provided, `''` is used. If `group_num` is not provided, and `flag` contains `'e`', `1` is used.
  If `group_num` is provided but the flag does not contain `e`, then
  it behaves as if it did. If the flag does contain `e`, then the location of one of
  the subgroups of the match is returned instead of the location of the
  entire match. The subgroup returned corresponds to the `group_num` argument
  (with 1-indexing).

  If there are an insufficient number of matches, or the pattern is empty,
  `0` is returned.

  - 2 arguments: Returns the index of the first `'#'` in `A`.

  ```sql
  SELECT REGEXP_INSTR(A, '#')
  ```

  - 3 arguments: Returns the starting index of the first occurrence of 3 consecutive digits in `A`,
    excluding the first 3 characters.

  ```sql
  SELECT REGEXP_INSTR(A, '\d{3}', 4)
  ```

  - 4 arguments: Returns the starting index of the 9th word sandwiched between angle brackets in `A`.

  ```sql
  SELECT REGEXP_INSTR(A, '<\w+>', 1, 9)
  ```

  - 5 arguments: Returns the ending index of the first substring of `A` that starts
    and ends with non-ascii characters.

  ```sql
  SELECT REGEXP_INSTR(A, '[^[:ascii:]].*[^[:ascii:]]', 1, 1, 1)
  ```

  - 6 arguments: Returns the starting index of the second line of `A` that begins with an uppercase vowel.

  ```sql
  SELECT REGEXP_INSTR(A, '^[AEIOU].*', 1, 2, 0, 'm')
  ```

  - 7 arguments: Looks for the first substring of `A` that has the format of a name in a phonebook (i.e. `Lastname, Firstname`)
    and returns the starting index of the first name.

  ```sql
  SELECT REGEXP_INSTR(A, '([[:upper]][[:lower:]]+), ([[:upper]][[:lower:]]+)', 1, 1, 0, 'e', 2)
  ```
