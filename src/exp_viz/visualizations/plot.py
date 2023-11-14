from abc import ABC

class Plot(ABC):
    """Abstract base class for all the visualization classes that generate plots and verbal explanations"""
    def __init__(self, model: object = None, explanation_type: str = 'global', instance_loc: int = None) -> None:
        self._model = model
        self._explanation_type = explanation_type
        self._instance_loc = instance_loc

    def generate_explanation(self, explanations):
        raise NotImplementedError
