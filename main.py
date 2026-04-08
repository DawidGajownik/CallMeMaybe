import shlex
from cmath import inf
from typing import List
import json
from llm_sdk.llm_sdk import Small_LLM_Model
from enum import Enum, auto

from objects.function import Function


class State(Enum):
    START = auto()
    PROMPT_KEY = auto()
    PROMPT_VALUE = auto()
    FUNCTION_KEY = auto()
    FUNCTION_VALUE = auto()
    ARGUMENTS_KEY = auto()
    ARGUMENT_OBJECT_KEY = auto()
    ARGUMENT_OBJECT_ARGUMENT = auto()
    DONE = auto()


def is_float(value):
    try:
        int(value)
        return False
    except ValueError:
        if value[-1] ==".":
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False


def choose_state(current_state: State, tokens, functions: List[Function], func, arg_finished, prompt: str) -> State:
    if current_state == State.START and "{" in tokens:
        return State.PROMPT_KEY
    elif current_state == State.PROMPT_KEY and "{\"prompt\": \"" in tokens:
        return State.PROMPT_VALUE
    elif current_state == State.PROMPT_VALUE and (f'"{prompt}",' in tokens or f'"{prompt.replace('"', '\\"')}",' in tokens):
        return State.FUNCTION_KEY
    elif current_state == State.FUNCTION_KEY and '", "name": "' in tokens:
        return State.FUNCTION_VALUE
    elif current_state == State.FUNCTION_VALUE and any(f"\"{fn.name}\"," in tokens for fn in functions):
        return State.ARGUMENTS_KEY
    elif current_state == State.ARGUMENTS_KEY and '", "parameters": {"' in tokens:
        return State.ARGUMENT_OBJECT_KEY
    elif current_state == State.ARGUMENT_OBJECT_KEY and f'"{func.get_actual_param()[0]}"{":" if func.get_actual_param()[1] == "string" else ""}' in tokens:
        return State.ARGUMENT_OBJECT_ARGUMENT
    elif current_state == State.ARGUMENT_OBJECT_ARGUMENT and (is_float(tokens.split()[-1]) or arg_finished):
        if func.increment_param_and_is_last():
            func.reset_param()
            return State.DONE
        return State.ARGUMENT_OBJECT_KEY
    elif current_state == State.DONE and "}}" in tokens:
        return State.START
    else:
        return current_state


def find_best(logits, amount):
    result = []
    for i in range(len(logits)):
        if len(result) < amount:
            result.append((logits[i],i))
        else:
            if logits[i] > result[-1][0]:
                result.pop(-1)
                result.append((logits[i],i))
        result.sort(key=lambda x: x[0], reverse=True)
    return result


def build_prompt(functions: list[Function]) -> str:
    prompt = "Available functions:\n"
    for fn in functions:
        prompt += fn.to_prompt() + "\n"
    prompt += "\nReturn ONLY valid JSON in the format:\n"
    prompt += '{"prompt": "...", "name": "...", "parameters": { ... } }\n'
    prompt += 'for example:\n'
    prompt += 'User: "What is the sum of 2 and 3"\n'
    prompt += 'Output: {"prompt": "What is the sum of 2 and 3", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}'
    return prompt


def get_params(function):
    params = []
    for param in function.params:
        params.append(param)
    return params


def set_function(functions, tokens):
    name = tokens.split('"name": ')[1].split(',')[0].strip('"')
    for fn in functions:
        if fn.name == name:
            return fn
    return None


def allowed_tokens(state, functions, selected_function, test: str):
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
        #print(selected_function.params)
        result = [selected_function.get_actual_param()[0]+"\":, \""]
        #print(result)
        return result
    if state == State.ARGUMENT_OBJECT_ARGUMENT and selected_function:
        choice = selected_function.get_actual_param()[1]
        nums = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '.', ' ', '-', ' -'
        ]
        if selected_function.is_last_param():
            nums.extend(['}', '}}'])
        return nums if choice == "number" else shlex.split(test)+['"', ", "] if choice == "string" else None
    if state == State.DONE:
        return ["}"]
    return []


def print_fn(functions):
    for function in functions:
        print("Name:", function["name"])
        print("Description:", function["description"])
        parameters = function["parameters"]
        for parameter in parameters:
            print("Parameter:", parameter)
            print("Type:", parameters[parameter]["type"])


def tokens_dict():
    return {
        State.START: [],
        State.PROMPT_KEY: [],
        State.PROMPT_VALUE: [],
        State.FUNCTION_KEY: [],
        State.FUNCTION_VALUE: [],
        State.ARGUMENTS_KEY: [],
        State.ARGUMENT_OBJECT_KEY: {},
        State.ARGUMENT_OBJECT_ARGUMENT: {},
        State.DONE: []
    }


def argument_finished(selected_fn, extra_tokens_dict, llm, test):
    if selected_fn is not None:
        key = selected_fn.get_actual_param()

        if key[0] in extra_tokens_dict[State.ARGUMENT_OBJECT_ARGUMENT]:
            decoded = llm.decode(
                extra_tokens_dict[State.ARGUMENT_OBJECT_ARGUMENT][key[0]]
            ).strip()
            if decoded.endswith('"') and len(decoded)>4 and selected_fn.is_last_param():
                return True
            if decoded.endswith('"') and len(decoded)>4 and not selected_fn.is_last_param():
                return True

    return False


def main():
    llm = Small_LLM_Model()
    tests = json.load(open('data/input/function_calling_tests.json'))
    functions = json.load(open('data/input/functions_definition.json'))
    functions_list = [Function.create(f) for f in functions]
    prompts = build_prompt(functions_list)
    results = []
    for test in tests:
        for function in functions_list:
            function.reset_param()
        prompt = prompts + f"\nUser: {test.get('prompt')}\nOutput:"
        tokens: List = llm.encode(prompt)[0].tolist()
        tokens.append(llm.encode("\n")[0].tolist()[0])
        extra_tokens_dict = tokens_dict()
        extra_tokens = []
        i = 0
        state = State.START
        selected_fn = None
        while i < 100:
            logits = llm.get_logits_from_input_ids(tokens+extra_tokens)
            allowed = allowed_tokens(state, functions_list, selected_fn, test.get('prompt'))
            masked_logits = [-inf] * len(logits)
            for item in allowed:
                for idx in llm.encode(item).tolist()[0]:
                    if state == State.PROMPT_VALUE:
                        indx = llm.decode(idx)
                        pr: str = test.get('prompt')
                        tok = llm.decode(extra_tokens_dict[state]).replace('\\"', '"')
                        if (pr.removeprefix(tok)+'",').startswith(indx):
                            masked_logits[idx] = logits[idx]
                    else:
                        masked_logits[idx] = logits[idx]
            best = find_best(
                logits if state == State.ARGUMENT_OBJECT_ARGUMENT and selected_fn.get_actual_param()[1] == "string" else masked_logits, 3
            )
            for logit in best[:1]:
                token = logit[1]
                if state in [State.ARGUMENT_OBJECT_KEY, State.ARGUMENT_OBJECT_ARGUMENT] and selected_fn:
                    param_name = selected_fn.get_actual_param()[0]
                    if param_name not in extra_tokens_dict[state]:
                        extra_tokens_dict[state][param_name] = []
                    extra_tokens_dict[state][param_name].append(token)
                else:
                    extra_tokens_dict[state].append(token)
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

                        if test.get('prompt') == llm.decode(extra_tokens_dict[state_key]):
                            tokens_after = llm.encode(test.get('prompt').replace('"', '\\"'))
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
                i += 1
                if "}}" in llm.decode(extra_tokens):
                    i = 100
            if state == State.ARGUMENTS_KEY:
                selected_fn = set_function(functions_list, llm.decode(extra_tokens))
            print(state, llm.decode(extra_tokens))
            state = choose_state(state, llm.decode(extra_tokens), functions_list, selected_fn, argument_finished(selected_fn, extra_tokens_dict, llm, test), test.get('prompt'))
        try:
            results.append(json.loads(llm.decode(extra_tokens).strip('\n')))
        except json.decoder.JSONDecodeError as e:
            print(e)
    with open("outputgood.json", "w") as f:
        json.dump(results, f, indent=2)


main()