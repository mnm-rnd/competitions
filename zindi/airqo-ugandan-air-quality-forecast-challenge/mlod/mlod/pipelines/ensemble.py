from typing import Tuple, List
from mlod.preprocessors import PreProcessor
from mlod.models import Model

from .pipeline import Pipeline

class EnsemblePipeline(Pipeline):
    def __init__(self, 
                 config_type: str, 
                 *model_preprocessor: List[Tuple[Model, PreProcessor]]):
        pass

class BoostEnsemblePipeline(EnsemblePipeline):
    pass