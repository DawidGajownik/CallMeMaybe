from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, field_validator


class Function(BaseModel):
    name: str
    desc: str
    params: List[Tuple[str, str]]

    # pola pomocnicze (nie wejściowe)
    param_counter: int = 0
    param: int = 0

    def model_post_init(self, __context):
        self.param_counter = len(self.params) - 1
        self.param = 0

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: List[Tuple[str, str]]):
        allowed_types = {"number", "string"}
        for name, param_type in value:
            if param_type not in allowed_types:
                raise ValueError(f"{param_type} is not a proper parameter type")
        return value

    def get_actual_param(self):
        return self.params[self.param]

    def reset_param(self):
        self.param = 0

    def increment_param_and_is_last(self) -> bool:
        if self.is_last_param():
            return True
        self.param += 1
        return False

    def is_last_param(self) -> bool:
        return self.param == self.param_counter

    @classmethod
    def create(cls, line: Dict) -> "Function":
        name = line["name"]
        desc = line["description"]
        parameters = line["parameters"]
        params = []
        for parameter in parameters:
            param_type = parameters[parameter]["type"]
            params.append((parameter, param_type))
        return cls(name=name, desc=desc, params=params)

    def to_prompt(self) -> str:
        params_str = ", ".join(f"{k}: {v}" for k, v in self.params)
        return f"- {self.name} - description = ({self.desc}), parameters = ({params_str})"

    def __str__(self):
        return f"Function: Name: {self.name} Desc: {self.desc} Params: {self.params}"