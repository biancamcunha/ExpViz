class Explainer():
    """Base class for the classes that will display the explanation visualizations using different explanation methods."""

    def _generate_explanation(self, features):
        raise NotImplementedError

    def _generate_explanation_plots(self, explanations, *args, **kwargs):
        raise NotImplementedError

    def generate_explanation_visualizations(self):
        """Public method that generates the desired explanations visualizations."""
        raise NotImplementedError