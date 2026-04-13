from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, field_validator


class Function(BaseModel):
    """model representing function"""
    name: str
    desc: str
    params: List[Tuple[str, str]]
    param_counter: int = 0
    param: int = 0

    def get_params(self) -> List:
        """return list of params"""
        params = []
        for param in self.params:
            params.append(param)
        return params

    def model_post_init(self, __context: Any) -> None:
        """post initialization parameters"""
        self.param_counter = len(self.params) - 1
        self.param = 0

    @field_validator("params")
    @classmethod
    def validate_params(
            cls, value: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """validate params value"""
        allowed_types = {"number", "string", "boolean"}
        for name, param_type in value:
            if param_type not in allowed_types:
                raise ValueError(
                    f"{param_type} is not a proper parameter type")
        return value

    def get_actual_param(self) -> Tuple[str, str]:
        """returning param actually processed in code"""
        return self.params[self.param]

    def reset_param(self) -> None:
        """reset processed param index"""
        self.param = 0

    def increment_param_and_is_last(self) -> bool:
        """
        returning true if param is last,
        if not increasing param and returning false
        """
        if self.is_last_param():
            return True
        self.param += 1
        return False

    def is_last_param(self) -> bool:
        """return true if param is last"""
        return self.param == self.param_counter

    @classmethod
    def create(cls, line: Dict) -> "Function":
        """create function from JSON"""
        name = line["name"]
        desc = line["description"]
        parameters = line["parameters"]
        params = []
        for parameter in parameters:
            param_type = parameters[parameter]["type"]
            params.append((parameter, param_type))
        return cls(name=name, desc=desc, params=params)

    def to_prompt(self) -> str:
        """function to prompt """
        params_str = ", ".join(f"{k}: {v}" for k, v in self.params)
        return (f"- {self.name} - description = ({self.desc})"
                f", parameters = ({params_str})")

    def __str__(self) -> str:
        """return string representation of function"""
        return (f"Function: Name: {self.name} "
                f"Desc: {self.desc} Params: {self.params}")
