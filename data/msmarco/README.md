# MS MARCO Dataset

## Data Splits

| Name | Train | Validation | Test |
|------|-------|------------|------|
| v1.1 | 82,326 | 10,047 | 9,650 |
| v2.1 | 808,731 | 101,093 | 101,092 |

## Data Fields

The data fields are the same among all splits (v1.1 and v2.1):

- `answers`: a list of string features
- `passages`: a dictionary feature containing:
  - `is_selected`: an int32 feature
  - `passage_text`: a string feature
  - `url`: a string feature
- `query`: a string feature
- `query_id`: an int32 feature
- `query_type`: a string feature
- `wellFormedAnswers`: a list of string features