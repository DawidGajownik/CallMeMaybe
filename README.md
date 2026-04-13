*This project has been created as part of the 42 curriculum by dawid-gajownik.*

# Call Me Maybe — Introduction to function calling in LLMs

## Description
This project implements a constrained-decoding pipeline for small LLMs to reliably produce structured function-call outputs (JSON with typed arguments). The goal is to bridge natural-language requests and machine-executable calls while keeping the implementation robust, testable, and reproducible.

## Instructions
- Requirements: Python 3.10+, dependencies in pyproject.toml
- Install: python -m pip install --user -r requirements.txt (or use poetry/uv where provided)
- Run (example):

  python -m src --functions_definition path/to/functions.json --input data/input/example.json --output output/function_calling_results.json

- Tests / quick check:
  - Ensure sample inputs are in data/input/
  - Expected output file: output/function_calling_results.json

## Algorithm explanation (Constrained decoding)
Constrained decoding guides the model token-by-token using a deterministic automaton that enforces the JSON schema and allowed tokens for each function argument. At every generation step the decoder prunes candidates that would violate the grammar (brackets, commas, types) or argument typing rules. This reduces malformed outputs and ensures generated function calls are syntactically valid without post-hoc parsing hacks.

Key points:
- A JSON grammar defines legal token sequences
- A finite-state machine (FSM) tracks progress through the expected structure
- Token scoring is masked to disallow illegal continuations
- Small LLM is used for token probabilities; decoding constraints guarantee correctness

## Design decisions
- Language: Python 3.10 for typing and compatibility
- Constrained-decoder implemented as a streaming validator + token filter for efficiency
- Strict schema validation used to fail-fast on semantic mismatches
- Flake8 enforced for code quality

## Performance analysis
- Accuracy: constrained decoding yields much higher structural validity (empirically >> prompt-only baseline)
- Speed: overhead from constraint checks is modest; decoding remains I/O / model-latency bound
- Reliability: deterministic FSM ensures invalid JSON is never produced, improving downstream robustness

## Challenges faced
- Small LLMs often produce partial or malformed JSON; solved by token-level masking and schema-driven constraints
- Handling ambiguous user prompts required prompt normalization and fallback heuristics
- Balancing strict constraints with model flexibility (e.g., string content) required careful token vocabulary decisions

## Testing strategy
- Unit tests for the decoding FSM and JSON validator
- Integration tests: sample inputs in data/input/ and end-to-end runs that compare output/function_calling_results.json against expected schemas
- Edge cases: empty inputs, wrong types, multi-parameter functions, nested structures

## Example usage
1. Place input files in data/input/ (examples provided).
2. Run:

   python -m src --functions_definition llm_sdk/functions.json --input data/input/example.json --output output/function_calling_results.json

3. Check output/function_calling_results.json for generated function calls.

## Resources
- Constrained decoding / constrained beam search literature
- OpenAI function-calling docs (example patterns)
- JSON Schema specification

## How AI was used
AI assisted in: drafting prompts, writing test prompts, and prototyping constrained-decoding ideas. All final code, tests, and documentation were written and verified by the author.


