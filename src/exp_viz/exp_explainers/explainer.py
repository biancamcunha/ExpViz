from abc import ABC
from pandas import DataFrame

class Explainer(ABC):
    """
    Base class for the classes that will generate and display the explanation visualizations using
    different explanation methods.
    """

    def _generate_explanation(self, features: DataFrame) -> any:
        raise NotImplementedError

    def _generate_explanation_plots(self, explanations, *args, **kwargs) -> None:
        raise NotImplementedError

    def generate_explanation_visualizations(self) -> None:
        """Public method that generates the desired explanations visualizations."""
        raise NotImplementedError

    @classmethod
    def list_visualization_options(cls) -> None:
        """Public class method that lists the possible plot names for the explainer."""
        raise NotImplementedError

    @classmethod
    def get_visualization_objective(cls, plot_name: str) -> None:
        """Public class method that displays the text that explains the chosen visualization
        objective."""
        raise NotImplementedError
