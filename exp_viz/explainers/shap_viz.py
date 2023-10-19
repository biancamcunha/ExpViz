import shap
from sklearn.base import is_classifier

class SHAPViz():
    """Class that generates explanations and explanation visualizations by using SHAP method."""

    def __init__(self, model, X, explanation_type = 'global', index = None) -> None:
        self.model = model
        self.X = X
        self.explanation_type = explanation_type
        self.index = index
        self.plot_options = {
            "summary_plot": shap.summary_plot,
            "decision_plot": shap.plots.decision,
            "bar_plot": shap.plots.bar,
            "waterfall_plot": shap.plots.waterfall,
            "dependence_plot": shap.dependence_plot,
            "force_plot": shap.force_plot
            }

    @staticmethod
    def _build_explainer(model, X):
        return shap.Explainer(model.predict, X)

    @staticmethod
    def _generate_explanations(explainer, X):
        return explainer(X)

    def _generate_global_explanations(self, plot_names, values):
        for plot in plot_names:
            if plot in self.plot_options:
                if is_classifier:
                    self.plot_options[plot](values[:, :, 0])
                else:
                    self.plot_options[plot](values)
            else:
                raise ValueError(f"The given plot name \"{plot}\" is not valid.")

    def _generate_local_explanations(self, plot_names, values, index):
        for plot in plot_names:
            if plot in self.plot_options:
                if is_classifier:
                    self.plot_options[plot](values[index, :, 0])
                else:
                    self.plot_options[plot](values[index])
            else:
                raise ValueError(f"The given plot name \"{plot}\" is not valid.")

    def generate_explanation_visualizations(self, plot_names=None):
        """Public method that generates the desired explanations visualizations."""
        explainer = self._build_explainer(self.model, self.X)
        shap_values = self._generate_explanations(explainer, self.X)
        if self.explanation_type == 'global':
            self._generate_global_explanations(plot_names, shap_values)
        else:
            self._generate_local_explanations(plot_names, shap_values, self.index)

