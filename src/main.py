import os
import shlex
import sys
import json
from argparse import Namespace
from cmath import inf
from typing import List, Any, Tuple, Dict
from llm_sdk.llm_sdk import Small_LLM_Model
from .objects import Function, State
from pathlib import Path
import argparse


def parse_args() -> Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json"),
    )

    return parser.parse_args()


def is_float(value: str) -> bool:
    """check if str is in a float format"""
    try:
        int(value)
        return False
    except ValueError:
        if value[-1] == ".":
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False


def start_ready(state: State, tokens: str) -> bool:
    """checks if start state is already created"""
    return state == State.START and "{" in tokens


def prompt_key_ready(state: State, tokens: str) -> bool:
    """checks if prompt key state is already created"""
    return state == State.PROMPT_KEY and "{\"prompt\": \"" in tokens


def prompt_value_ready(
        state: State, prompt: str, tokens: str
) -> bool:
    """checks if prompt value state is already created"""
    return (
            state == State.PROMPT_VALUE
            and (
                  f'"{prompt}",' in tokens or
                  f'"{prompt.replace('"', '\\"')}",' in tokens))


def function_key_ready(state: State, tokens: str) -> bool:
    """checks if function key state is already created"""
    return state == State.FUNCTION_KEY and '", "name": "' in tokens


def function_value_ready(
        state: State, tokens: str,
        functions: List[Function]) -> bool:
    """checks if function value state is already created"""
    return (
            state == State.FUNCTION_VALUE
            and any(f"\"{fn.name}\"," in tokens for fn in functions))


def argument_key_ready(state: State, tokens: str) -> bool:
    """checks if argument key state is already created"""
    return (
            state == State.ARGUMENTS_KEY
            and '", "parameters": {"' in tokens)


def arg_obj_key_ready(
        state: State, tokens: str,
        func: Function | None) -> bool:
    """checks if argument object key state is already created"""
    return (
            state == State.ARGUMENT_OBJECT_KEY
            and func is not None
            and (
                    f'"{func.get_actual_param()[0]}"'
                    f'{":" if func.get_actual_param()[1] == "string" else ""}'
            ) in tokens
    )


def arg_obj_val_ready(
        state: State,
        tokens: str,
        func: Function | None,
        arg_finished: bool,
        arg_match_bool: bool
) -> bool:
    """checks if argument object value state is already created"""
    return (
            state == State.ARGUMENT_OBJECT_ARGUMENT
            and (
                    is_float(tokens.split()[-1])
                    or arg_finished or arg_match_bool
            )
            and func is not None
    )


def all_args_ready(func: Function | None) -> bool:
    """checks if all arguments are created"""
    if func is not None:
        if func.increment_param_and_is_last():
            func.reset_param()
            return True
    return False


def done_ready(state: State, tokens: str) -> bool:
    """checks if json is finished"""
    return state == State.DONE and "}}" in tokens


def choose_state(
        current_state: State,
        tokens: str,
        functions: List[Function],
        func: Function | None,
        arg_finished: bool,
        arg_match_bool: bool,
        prompt: str) -> State:
    """chooses state based on prompt value"""
    if start_ready(current_state, tokens):
        return State.PROMPT_KEY
    elif prompt_key_ready(current_state, tokens):
        return State.PROMPT_VALUE
    elif prompt_value_ready(current_state, prompt, tokens):
        return State.FUNCTION_KEY
    elif function_key_ready(current_state, tokens):
        return State.FUNCTION_VALUE
    elif function_value_ready(current_state, tokens, functions):
        return State.ARGUMENTS_KEY
    elif argument_key_ready(current_state, tokens):
        return State.ARGUMENT_OBJECT_KEY
    elif arg_obj_key_ready(current_state, tokens, func):
        return State.ARGUMENT_OBJECT_ARGUMENT
    elif arg_obj_val_ready(
            current_state, tokens, func,
            arg_finished, arg_match_bool):
        if all_args_ready(func):
            return State.START
        return State.ARGUMENT_OBJECT_KEY
    elif done_ready(current_state, tokens):
        return State.START
    else:
        return current_state


def find_best(
        logits: List[float], amount: int
) -> List[Tuple[float, int]]:
    """finds best parameters"""
    result: List = []
    for i in range(len(logits)):
        if len(result) < amount:
            result.append((logits[i], i))
        else:
            if logits[i] > result[-1][0]:
                result.pop(-1)
                result.append((logits[i], i))
        result.sort(key=lambda x: x[0], reverse=True)
    return result


def build_prompt(functions: list[Function]) -> str:
    """building standard prompt"""
    prompt = "Available functions:\n"
    for fn in functions:
        prompt += fn.to_prompt() + "\n"
    prompt += "\nReturn ONLY valid JSON in the format:\n"
    prompt += '{"prompt": "...", "name": "...", "parameters": { ... } }\n'
    prompt += 'for example:\n'
    prompt += 'User: "What is the sum of 2 and 3"\n'
    prompt += ('Output: {"prompt": "What is the sum of 2 and 3", '
               '"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}')
    return prompt


def set_function(functions: List[Function], tokens: str) -> Function | None:
    """setting the detected function"""
    name = tokens.split('"name": ')[1].split(',')[0].strip('"')
    for fn in functions:
        if fn.name == name:
            return fn
    return None


def argument_tokens(func: Function, test: str) -> List[str]:
    """returning correct tokens for argument"""
    choice = func.get_actual_param()[1]
    nums = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '.', ' ', '-', ' -'
    ]
    if func.is_last_param():
        nums.extend(['}', '}}'])
    if choice not in ["number", "string", "boolean"]:
        raise ValueError(choice, "is not a proper argument type.")
    return (
        nums
        if choice == "number"
        else shlex.split(test) + ['"', ", "]
        if choice == "string"
        else ['true', 'false']
    )


def allowed_tokens(
        state: State, functions: List[Function],
        selected_function: Function | None, test: str) -> List[str]:
    """returning correct tokens depending on state"""
    if state == State.START:
        return ["{\""]
    if state == State.PROMPT_KEY:
        return ["\"prompt\": \""]
    if state == State.PROMPT_VALUE:
        return [f'{test}", ']
    if state == State.FUNCTION_KEY:
        return ["name\": \""]
    if state == State.FUNCTION_VALUE:
        return [f'"{fn.name}",' for fn in functions]
    if state == State.ARGUMENTS_KEY:
        return [', "parameters": {"']
    if state == State.ARGUMENT_OBJECT_KEY and selected_function:
        result = [selected_function.get_actual_param()[0]+"\":, \""]
        return result
    if state == State.ARGUMENT_OBJECT_ARGUMENT and selected_function:
        return argument_tokens(selected_function, test)
    if state == State.DONE:
        return ["}"]
    return []


def tokens_dict(prompt: str, llm: Small_LLM_Model) -> Dict:
    """returns empty tokens dict"""
    return {
        State.START: llm.encode("{\"").tolist()[0],
        State.PROMPT_KEY: llm.encode("prompt\": \"").tolist()[0],
        State.PROMPT_VALUE: llm.encode(f'{prompt}", ').tolist()[0],
        State.FUNCTION_KEY: llm.encode("\"name\": \"").tolist()[0],
        State.FUNCTION_VALUE: [],
        State.ARGUMENTS_KEY: [],
        State.ARGUMENT_OBJECT_KEY: {},
        State.ARGUMENT_OBJECT_ARGUMENT: {},
        State.DONE: []
    }


def arg_matches_bool(
        selected_fn: Function | None,
        extra_tokens_dict: Dict, llm: Small_LLM_Model) -> bool:
    """checks if argument is correct boolean"""
    if selected_fn is not None:
        key = selected_fn.get_actual_param()
        if key[0] in extra_tokens_dict[State.ARGUMENT_OBJECT_ARGUMENT]:
            decoded = llm.decode(
                extra_tokens_dict[State.ARGUMENT_OBJECT_ARGUMENT][key[0]]
            ).strip()
            if decoded in ["true", "false"]:
                return True
    return False


def argument_finished(
        selected_fn: Function | None, extra_tokens_dict: Dict,
        llm: Small_LLM_Model) -> bool:
    """checks if argument is closed with double quote"""
    if selected_fn is not None:
        key = selected_fn.get_actual_param()
        if key[0] in extra_tokens_dict[State.ARGUMENT_OBJECT_ARGUMENT]:
            decoded = llm.decode(
                extra_tokens_dict[State.ARGUMENT_OBJECT_ARGUMENT][key[0]]
            ).strip()
            if decoded.endswith('"') and len(decoded) > 4:
                return True
    return False


def build_final_prompt(main_prompt: str, prompt: str) -> str:
    "returning final prompt sent to llm"
    return main_prompt + f"\nUser: {prompt}\nOutput:"


def final_prompt_to_tokens(
        llm: Small_LLM_Model, final_prompt: str
) -> List[int]:
    """final prompt to tokens translation"""
    tokens: List = llm.encode(final_prompt)[0].tolist()
    tokens.append(llm.encode("\n")[0].tolist()[0])
    return tokens


def all_logits_available(state: State, fn: Function) -> bool:
    """returning true if argument is string"""
    return (state == State.ARGUMENT_OBJECT_ARGUMENT
            and fn.get_actual_param()[1] == "string")


def get_best_token(
        llm: Small_LLM_Model,
        state: State,
        functions: list,
        selected_fn: Function | None,
        prompt: str,
        logits: list,
        extra_tokens_dict: dict
) -> int:
    """returning single best token"""
    allowed = allowed_tokens(state, functions, selected_fn, prompt)
    masked_logits = [-inf] * len(logits)
    for item in allowed:
        for idx in llm.encode(item).tolist()[0]:
            if state in [State.PROMPT_VALUE]:
                index = llm.decode(idx)
                pr: str = prompt
                tok = llm.decode(extra_tokens_dict[state]).replace('\\"', '"')
                if (pr.removeprefix(tok) + '",').startswith(index):
                    masked_logits[idx] = logits[idx]
            else:
                masked_logits[idx] = logits[idx]
    best = find_best(
        logits
        if selected_fn and all_logits_available(state, selected_fn)
        else masked_logits, 1
    )
    return best[0][1]


def add_token_to_dict(
        state: State, fn: Function | None, token: int,
        extra_tokens_dict: dict) -> dict:
    """adding token to tokens dict"""
    if state in [
        State.ARGUMENT_OBJECT_KEY, State.ARGUMENT_OBJECT_ARGUMENT
    ] and fn:
        param_name = fn.get_actual_param()[0]
        if param_name not in extra_tokens_dict[state]:
            extra_tokens_dict[state][param_name] = []
        extra_tokens_dict[state][param_name].append(token)
    else:
        extra_tokens_dict[state].append(token)
    return extra_tokens_dict


def tokens_dict_to_list(
        llm: Small_LLM_Model, prompt: str,
        extra_tokens_dict: dict) -> List[int]:
    """converting tokens dict into tokens list"""
    extra_tokens = []
    ordered_states = [
        State.START,
        State.PROMPT_KEY,
        State.PROMPT_VALUE,
        State.FUNCTION_KEY,
        State.FUNCTION_VALUE,
        State.ARGUMENTS_KEY
    ]
    for state_key in ordered_states:
        if state_key == State.PROMPT_VALUE:
            if prompt == llm.decode(extra_tokens_dict[state_key]):
                tokens_after = llm.encode(prompt.replace('"', '\\"'))
                extra_tokens_dict[state_key] = tokens_after.tolist()[0]
        extra_tokens.extend(extra_tokens_dict[state_key])
    keys_dict = extra_tokens_dict[State.ARGUMENT_OBJECT_KEY]
    values_dict = extra_tokens_dict[State.ARGUMENT_OBJECT_ARGUMENT]
    keys_order = list(keys_dict.keys())
    values_order = list(values_dict.keys())
    max_len = max(len(keys_order), len(values_order))
    for j in range(max_len):
        if j < len(keys_order):
            key = keys_order[j]
            extra_tokens.extend(keys_dict[key])
        if j < len(values_order):
            key = values_order[j]
            extra_tokens.extend(values_dict[key])
    extra_tokens.extend(extra_tokens_dict[State.DONE])
    return extra_tokens


def functions_from_json(functions: Any) -> List[Function]:
    """iterating over JSON to create function objects"""
    return [
        Function.create(f) for f in functions
        if isinstance(Function.create(f), Function)
    ]


def init(
        args: Namespace
) -> Tuple[Any, List[Function], str, Small_LLM_Model, List]:
    """initialization"""
    try:
        with open(args.input) as f:
            calls = json.load(f)
        with open(args.functions_definition, 'r') as f:
            functions_json = json.load(f)
        functions = functions_from_json(functions_json)
        main_prompt = build_prompt(functions)
        llm = Small_LLM_Model()
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Missing key: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Invalid value: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    output: List = []
    return calls, functions, main_prompt, llm, output


def reset_functions_params_counter(functions: List[Function]) -> None:
    """iterating over functions to reset params counter"""
    for function in functions:
        function.reset_param()


def main() -> None:
    """
    main function
    1. initialization
    2. iterating over prompt calls to create list of JSONs for output
    3. export to JSON file
    """
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    calls, functions, main_prompt, llm, output = init(args)
    for call in calls:
        prompt = call.get('prompt')
        reset_functions_params_counter(functions)
        final_prompt = build_final_prompt(main_prompt, prompt)
        tokens = final_prompt_to_tokens(llm, final_prompt)
        extra_tokens_dict = tokens_dict(prompt, llm)
        extra_tokens = tokens_dict_to_list(
            llm, prompt, extra_tokens_dict)
        i = 0
        state = State.FUNCTION_VALUE
        selected_fn: Function | None = None
        while i < 150:
            logits = llm.get_logits_from_input_ids(tokens+extra_tokens)
            try:
                token = get_best_token(
                    llm, state, functions, selected_fn,
                    prompt, logits, extra_tokens_dict)
                extra_tokens_dict = add_token_to_dict(
                    state, selected_fn, token, extra_tokens_dict)
                extra_tokens = tokens_dict_to_list(
                    llm, prompt, extra_tokens_dict)
                i += 1
                if "}}" in llm.decode(extra_tokens):
                    i = 150
            except ValueError as e:
                print(e)
            if state == State.ARGUMENTS_KEY:
                selected_fn = set_function(functions, llm.decode(extra_tokens))
            print(state, llm.decode(extra_tokens))
            state = choose_state(
                state, llm.decode(extra_tokens), functions, selected_fn,
                argument_finished(selected_fn, extra_tokens_dict, llm),
                arg_matches_bool(selected_fn, extra_tokens_dict, llm),
                prompt)
            if state == State.START:
                if not done_ready(state, llm.decode(extra_tokens)):
                    extra_tokens_dict[State.DONE] = llm.encode("}}").tolist()[0]
                    extra_tokens = tokens_dict_to_list(
                        llm, prompt, extra_tokens_dict)
                state = State.FUNCTION_VALUE
                i = 150
        try:
            output.append(json.loads(llm.decode(extra_tokens).strip('\n')))
        except json.decoder.JSONDecodeError as e:
            print(e)
    os.makedirs("data/output", exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
