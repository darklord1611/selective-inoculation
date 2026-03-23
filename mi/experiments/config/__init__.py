from . import mixture_of_propensities
from . import selective_inoculation
from . import unsupervised_selective_inoculation

from abc import ABC
from mi.experiments.data_models import ExperimentConfig

class ConfigModule(ABC):
    def list_configs(self) -> list[ExperimentConfig]:
        pass
