from enum import Enum, auto


class ExplanationMethod(Enum):
    NONE = auto()  # Needed for sanity checks etc.
    INTEGRATED_GRADIENTS = auto()
    FREQ_INTEGRATED_GRADIENTS = auto()
    GRAD_CAM = auto()
    GRADIENT_SHAP = auto()
    OCCLUSION = auto()
    SAILIENCY = auto()

    @property
    def pretty_name(self) -> str:
        return self.name.replace("_", " ").title()
    
    def __repr__(self) -> str:
        return str(self).split(".")[-1]
