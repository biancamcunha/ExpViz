import shap
from sklearn.base import is_classifier

class SHAPViz():
    """Class that generates explanations and explanation visualizations by using SHAP method."""

    def __init__(self, model, features, explanation_type = 'global', instance_loc = None) -> None:
        self.explainer : shap.Explainer = shap.Explainer(model)
        self.model = model
        self.features = features
        self.explanation_type = explanation_type
        self.instance_loc = instance_loc
        self.local_plot_options = {
            "bar_plot": shap.plots.bar,
            "decision_plot": shap.decision_plot,
            "waterfall_plot": shap.plots.waterfall
            }
        self.global_plot_options = {
            "bar_plot": shap.plots.bar,
            "decision_plot": shap.decision_plot,
            "summary_plot": shap.summary_plot
            }

    def _generate_explanations(self, features):
        return self.explainer(features)

    def _generate_global_explanations(self, plot_names, values):
        for plot in plot_names:
            if plot in self.global_plot_options:
                if is_classifier(self.model):
                    if plot != 'decision_plot':
                        self.global_plot_options[plot](values[:, :, 0])
                    else:
                        self.global_plot_options[plot](self.explainer.expected_value[0],
                                                       self.explainer.shap_values(self.features)[0],
                                                       self.features.columns)
                else:
                    if plot != 'decision_plot':
                        self.global_plot_options[plot](values)
                    else:
                        self.global_plot_options[plot](self.explainer.expected_value,
                                                       self.explainer.shap_values(self.features),
                                                       self.features.columns)
            else:
                raise ValueError(f"The given plot name \"{plot}\" is not valid.")

    def _generate_local_explanations(self, plot_names, values, index):
        for plot in plot_names:
            if plot in self.local_plot_options:
                    if is_classifier(self.model):
                        if plot != 'decision_plot':
                            self.local_plot_options[plot](values[index, :, 0])
                        else:
                            self.local_plot_options[plot](self.explainer.expected_value[0],
                                                          self.explainer.shap_values(self.features)[0][index],
                                                          self.features.columns)
                    else:
                        if plot != 'decision_plot':
                            self.local_plot_options[plot](values[index])
                        else:
                            self.local_plot_options[plot](self.explainer.expected_value[0],
                                                          self.explainer.shap_values(self.features)[0][index],
                                                          self.features.columns)
            else:
                raise ValueError(f"The given plot name \"{plot}\" is not valid.")

    def generate_explanation_visualizations(self, plot_names=None):
        """Public method that generates the desired explanations visualizations."""
        if plot_names is None:
            if self.explanation_type == 'global':
                plot_names = self.global_plot_options
            else:
                plot_names = self.local_plot_options
        shap_values = self._generate_explanations(self.features)
        if self.explanation_type == 'global':
            self._generate_global_explanations(plot_names, shap_values)
        else:
            self._generate_local_explanations(plot_names, shap_values, self.instance_loc)
