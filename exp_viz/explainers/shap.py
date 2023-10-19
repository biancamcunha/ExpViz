import shap
from . import Explainer

class SHAP(Explainer):
    """Class that generates explanations and explanation visualizations by using SHAP method."""

    def __init__(self, shap_explainer_name) -> None:
        super().__init__( )
        self.shap_explainer_name = shap_explainer_name
        self.shap_explainer_name_options = {'TreeExplainer': shap.TreeExplainer,
                                            'GradientExplainer': shap.GradientExplainer,
                                            'DeepExplainer': shap.DeepExplainer,
                                            'KernelExplainer': shap.KernelExplainer}
        self.plot_name_options = [shap.summary_plot,
                                  shap.plots.decision,
                                  shap.plots.bar,
                                  shap.plots.waterfall,
                                  shap.dependence_plot,
                                  shap.force_plot]

    def _build_explainer(self, model):
        return self.shap_explainer_name_options[self.shap_explainer_name](model)

    @staticmethod
    def _generate_explanations(explainer, X):
        return explainer(X)

    @staticmethod
    def _generate_global_visualizations(plot_names, values):
        for plot in plot_names:
            plot(values)

    @staticmethod
    def _generate_local_explanations(plot_names, values):
        for plot in plot_names:
            plot(values[0])


    def generate_explanation_visualizations(self, model, X):
        """Public method that generates the desired explanations visualizations."""
        explainer = self._build_explainer(model)
        shap_values = self._generate_explanations(explainer, X)
        plot_names = self.plot_name_options
        self._generate_global_visualizations(plot_names, shap_values)
