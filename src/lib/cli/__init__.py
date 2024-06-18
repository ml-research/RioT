from .forecasting_cli import make_cli as make_forecasting_cli
from .classification_cli import make_cli as make_classification_cli

__all__ = [
    "make_forecasting_cli",
    "make_classification_cli",
]
