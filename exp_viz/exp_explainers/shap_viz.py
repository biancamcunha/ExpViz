from pandas import DataFrame
import shap
from sklearn.base import is_classifier
from .explainer import Explainer

class SHAPViz(Explainer):
    """Class that generates explanation visualizations by using SHAP explanation method."""

    def __init__(self, model: object, features: DataFrame, explanation_type: str = 'global', instance_loc: int = None) -> None:
        self.__explainer : shap.Explainer = shap.Explainer(model)
        self.__model: object = model
        self.__features: DataFrame = features
        self.__explanation_type: str = explanation_type
        self.__instance_loc: int = instance_loc
        self.__local_plot_options: dict = {
            "bar_plot": shap.plots.bar,
            "decision_plot": shap.decision_plot,
            "waterfall_plot": shap.plots.waterfall
            }
        self.__global_plot_options: dict = {
            "bar_plot": shap.plots.bar,
            "decision_plot": shap.decision_plot,
            "summary_plot": shap.summary_plot
            }

    def _generate_explanation(self, features: DataFrame):
        explanations = self.__explainer(features)
        shap_values = self.__explainer.shap_values(features)
        return explanations, shap_values

    def __generate_global_explanation_plots(self, plot_names: list[str], explanations: any, shap_values: any):
        for plot in plot_names:
            if plot in self.__global_plot_options:
                if is_classifier(self.__model):
                    if plot != 'decision_plot':
                        self.__global_plot_options[plot](explanations[:, :, 1])
                    else:
                        self.__global_plot_options[plot](self.__explainer.expected_value[1],
                                                       shap_values[1],
                                                       self.__features.columns)
                else:
                    if plot != 'decision_plot':
                        self.__global_plot_options[plot](explanations, self.__features)
                    else:
                        self.__global_plot_options[plot](self.__explainer.expected_value,
                                                       shap_values,
                                                       self.__features.columns)
            else:
                raise ValueError(f"The given plot name \"{plot}\" is not valid.")

    def __generate_local_explanation_plots(self, plot_names: list[str], explanations: any, shap_values: any, index: int):
        for plot in plot_names:
            if plot in self.__local_plot_options:
                    if is_classifier(self.__model):
                        if plot != 'decision_plot':
                            self.__local_plot_options[plot](explanations[index, :, 1], self.__features)
                        else:
                            self.__local_plot_options[plot](self.__explainer.expected_value[1],
                                                          shap_values[1][index],
                                                          self.__features.columns)
                    else:
                        if plot != 'decision_plot':
                            self.__local_plot_options[plot](explanations[index])
                        else:
                            self.__local_plot_options[plot](self.__explainer.expected_value[1],
                                                          shap_values[1][index],
                                                          self.__features.columns)
            else:
                raise ValueError(f"The given plot name \"{plot}\" is not valid.")

    def _generate_explanation_plots(self, explanations: any, plot_names: list[str], shap_values: any):
        if self.__explanation_type == 'global':
            self.__generate_global_explanation_plots(plot_names, explanations, shap_values)
        else:
            self.__generate_local_explanation_plots(plot_names, explanations, shap_values, self.__instance_loc)

    def generate_explanation_visualizations(self, plot_names: list[str] = None):
        """Public method that generates the desired explanations visualizations."""
        if plot_names is None:
            if self.__explanation_type == 'global':
                plot_names = self.__global_plot_options
            else:
                plot_names = self.__local_plot_options
        explanations, shap_values = self._generate_explanation(self.__features)
        self._generate_explanation_plots(explanations, plot_names, shap_values)
