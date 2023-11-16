from abc import ABC

class Plot(ABC):
    """Abstract base class for all the visualization classes that generate plots and verbal
    explanations"""
    def __init__(self, model: object = None, explanation_type: str = 'global',
                 instance_loc: int = None) -> None:
        self._model: object = model
        self._explanation_type: str = explanation_type
        self._instance_loc: int = instance_loc

    @staticmethod
    def _get_textual_explanation():
        raise NotImplementedError

    def generate_explanation(self, explanations: any):
        """Method that generates the chosen visualization for the model or prediction explanation
        and a textual explanation of the visualization personalized for the input data."""
        raise NotImplementedError
