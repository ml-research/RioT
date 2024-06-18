from pathlib import Path
from typing import Callable, Literal, TypeVar
import torch
import torch.nn as nn
import os

import yaml

T = TypeVar("T")

def execute_activation_hook(model, layer: nn.Module, x: torch.Tensor, activation_tf: Callable[[torch.Tensor], T]) -> T:
    hook_output = {}
    hook = layer.register_forward_hook(lambda m, i, o: hook_output.update({"out": activation_tf(o)}))
    model(x)
    hook.remove()
    return hook_output["out"]

def parse_string_ranges(feedback_penalty_range: list[str]) -> list[range]:
    return [
                range(*map(int, range_str.split(":")))
                for range_str in feedback_penalty_range
            ]



RECREATE_CONFOUNDER_DATASET = "RECREATE_CONFOUNDER_DATASET"

def get_debug_dataset_creation() -> bool:
    return os.environ.get(RECREATE_CONFOUNDER_DATASET, "0") == "1"

def set_debug_dataset_creation(val: bool) -> None:
    print(f"Setting {RECREATE_CONFOUNDER_DATASET} to {val}")
    os.environ[RECREATE_CONFOUNDER_DATASET] = "1" if val else "0"



def parse_lambda(kind: Literal["freq", "time"], task: Literal["classification", "forecasting"], exp_config: str)-> int:
    with open(Path(os.getcwd()) / "src" / "configs" / "exp" / task / exp_config, "r") as file:
        return int(yaml.safe_load(file)["data"]["init_args"][f"lambda_{kind}"])