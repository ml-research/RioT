from enum import Enum, auto, Flag


class Confounder(Flag):
    """
    Enum for the different confounder modes.
    INSERT_START: Insert confounder at the start of the sequence
    INSERT_MIDDLE: Insert confounder in the middle of the sequence
    NO_CONFOUNDER: No confounder

    """

    NO_CONFOUNDER = auto()
    INSERT_START = auto()
    INSERT_MIDDLE = auto()
    INSERT_SPLIT_4 = auto()
    MOVING_START = auto()
    KNOWN_START = auto()
    TARGET = auto()
    COVARIATE = auto()
    SINGLE = auto()
    SANITY = auto()

    FORECASTING_TIME = auto()
    FORECASTING_NOISE = auto()
    FORECASTING_DIRAC = auto()
    CLASSIFICATION_TIME = auto()
    CLASSIFICATION_FREQ = auto()

    @property
    def pretty_name(self) -> str:
        return str(self).replace("_", " ").title()

    @property
    def is_sanity_check(self) -> bool:
        return Confounder.SANITY in self

    @property
    def is_freq_confounder(self) -> bool:
        return Confounder.INSERT_SPLIT_4 in self

    @property
    def is_multivariate(self) -> bool:
        return Confounder.TARGET in self or Confounder.COVARIATE in self

    @property
    def is_only_covariate(self) -> bool:
        return (
            Confounder.COVARIATE in self
            and Confounder.TARGET not in self
            and Confounder.SINGLE not in self
        )

    @property
    def is_only_target(self) -> bool:
        return (
            Confounder.COVARIATE not in self
            and Confounder.TARGET in self
            and Confounder.SINGLE not in self
        )

    @property
    def is_single_covariate(self) -> bool:
        return (
            Confounder.COVARIATE in self
            and Confounder.SINGLE in self
            and Confounder.TARGET not in self
        )

    @property
    def is_no_confounder(self) -> bool:
        return Confounder.NO_CONFOUNDER == self

    def __repr__(self) -> str:
        return str(self).split(".")[-1]

    # @property
    # def is_confounder_only(self) -> bool:
    #     return Confounder.CONFOUNDER_ONLY in self


class ForecastingMode(Enum):
    UNIVARIATE = auto()
    MULTIVARIATE_UNIVARIATE = auto()
    MULTIVARIATE_MULTIVARIATE = auto()

    def __repr__(self) -> str:
        return str(self).split(".")[-1]


class P2SFeedbackMode(Enum):
    NO_FEEDBACK = auto()
    FULL_FEEDBACK = auto()
    LIMITED_FEEDBACK = auto()

    def __repr__(self) -> str:
        return str(self).split(".")[-1]