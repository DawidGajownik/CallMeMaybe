from typing import Dict, List


class Function:
    def __init__(self, name: str, desc: str, params: List):
        self.name = name
        self.desc = desc
        self.params = params
        self.param_counter = len(self.params) - 1
        self.param = 0

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
            params.append((parameter, parameters[parameter]["type"]))
            #params[parameter] = parameters[parameter]["type"]
        return cls(name, desc, params)

    def to_prompt(self) -> str:
        params_str = ", ".join(f"{k}: {v}" for k, v in self.params)
        return f"- {self.name} - description = ({self.desc}), parameters = ({params_str})"

    def __str__(self):
        return f"Function: Name: {self.name} Desc: {self.desc} Params: {self.params}"