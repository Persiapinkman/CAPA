# Prompt JSON Format (Input Normalization)

Prompt file must be valid JSON with two keys:

## `prompts`

Object mapping **class key** → list of prompt tokens (words for RexOmni to detect).

- Key: class name used in `index2cls`
- Value: list of strings, e.g. `["banner", "slogan"]` for one class

Example:

```json
"prompts": {
    "banner_slogan": ["banner", "slogan"],
    "suitcase": ["suitcase"]
}
```

## `index2cls`

Object mapping **category index (string)** → class key. Used for COCO `category_id` and ordering.

- Key: index as string, e.g. `"1"`, `"2"`
- Value: class key that must exist in `prompts`

Example:

```json
"index2cls": {
    "1": "banner_slogan",
    "2": "suitcase"
}
```

## Full Example

See [prompt_example.json](prompt_example.json).

Detection text is built in index order: for each index, the tokens from `prompts[index2cls[idx]]` are joined (e.g. "banner,slogan") and all are combined into one line: "Detect banner,slogan,suitcase. Output the bounding box coordinates in [x0, y0, x1, y1] format."
