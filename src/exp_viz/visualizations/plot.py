from abc import ABC

class Plot(ABC):
    """Abstract base class for all the visualization classes that generate plots and textual
    explanations"""
    def __init__(self, model: object = None, explanation_type: str = 'global',
                 instance_loc: int = None) -> None:
        self._model: object = model
        self._explanation_type: str = explanation_type
        self._instance_loc: int = instance_loc

    def _display_textual_explanation(self, explanations: any) -> None:
        raise NotImplementedError

    def generate_explanation(self, explanations: any) -> None:
        """Method that generates the chosen visualization for the model or prediction explanation
        and a textual explanation of the visualization personalized for the input data."""
        raise NotImplementedError
