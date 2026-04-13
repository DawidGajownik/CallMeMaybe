from enum import Enum, auto


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
