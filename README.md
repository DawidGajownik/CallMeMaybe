*This project has been created as part of the 42 curriculum by dgajowni.*

# Call Me Maybe — Introduction to function calling in LLMs

## Description
The aim of the project is to handle LLM in such a way that it always returns correct JSON calling functions.

## Instructions
- Requirements: Python 3.10+, dependencies in pyproject.toml
- Install: make install
- Run (example):
  
  make run
  python -m src --functions_definition path/to/functions.json --input data/input/example.json --output output/function_calling_results.json

- Tests / quick check:
  - Ensure sample inputs are in data/input/
  - Expected output file: output/function_calling_results.json

## Algorithm explanation (Constrained decoding)
My implementation divides the output creation into states. Each state allows for appropriate tokens. The program checks at each stage whether the state has been completed and changes it to the appropriate state.

## Design decisions
- Language: Python 3.10 for typing and compatibility
- mypy to check type hinting
- Flake8 enforced for code quality

## Performance analysis
- Accuracy: constrained decoding yields much higher structural validity
- Speed: overhead from constraint checks is modest
- Reliability: output can containn only correct JSON

## Challenges faced
Small models often produce incomplete or broken JSON → we fixed this by blocking invalid tokens and enforcing a schema
Ambiguous user prompts were an issue → we added prompt normalization and simple fallback rules
It was tricky to balance strict rules with model flexibility (like handling free text) → this required careful choice of allowed tokens

## Testing strategy
- Integration tests: sample inputs in data/input/ and end-to-end runs that compare output/function_calling_results.json against expected schemas
- Edge cases: empty inputs, wrong types, multi-parameter functions, nested structures

## Example usage
1. Place input files in data/input/ (examples provided).
2. Run:

   uv run python -m src

3. Check output/function_calling_results.json for generated function calls.

## Resources

### How AI was used
- Copilot/ChatGPT for drafting prompts, writing test prompts, and prototyping constrained-decoding ideas.


