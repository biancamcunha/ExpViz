import shap
import lime

class Explainer():
    """Explainer interface that lists the methods that need to be implemented in an explainer."""
    def __init__(self, method_names, explanation_types) -> None:
        self.explainer_names = method_names
        self.explanation_types = explanation_types
        self.explainer_name_options = {'shap': shap,
                                       'lime': lime}
        self.explanation_type_options = ['local', 'global']

    def _build_explainer(self, model):
        raise NotImplementedError

    @staticmethod
    def _generate_explanations(explainer, X):
        raise NotImplementedError

    @staticmethod
    def _generate_global_visualizations(plot_names, values):
        raise NotImplementedError

    @staticmethod
    def _generate_local_explanations(plot_names, values):
        raise NotImplementedError

    def generate_explanation_visualizations(self, model, X):
        raise NotImplementedError
        