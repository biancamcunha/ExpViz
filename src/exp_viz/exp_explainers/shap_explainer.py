from pandas import DataFrame
import shap
from .explainer import Explainer
from ..visualizations import SHAPBarPlot, SHAPBeeswarmPlot, SHAPWaterfallPlot

class SHAPExplainer(Explainer):
    """
    Class that generates explanations for Machine Learning models by using SHAP explanation
    method.
    """

    def __init__(self, model: object, features: DataFrame, explanation_type: str = 'global',
                 instance_loc: int = None) -> None:
        self.__model: object = model
        self.__explainer : shap.Explainer = shap.Explainer(self.__model)
        self.__features: DataFrame = features
        self.__explanation_type: str = explanation_type
        self.__instance_loc: int = instance_loc
        self.__local_plot_options: dict = {
            "bar_plot": SHAPBarPlot,
            "waterfall_plot": SHAPWaterfallPlot
            }
        self.__global_plot_options: dict = {
            "bar_plot": SHAPBarPlot,
            "beeswarm_plot": SHAPBeeswarmPlot
            }

    def _generate_explanation(self, features: DataFrame):
        explanations = self.__explainer(features)
        return explanations

    def _generate_explanation_plots(self, explanations: any, plot_names: list[str]):
        for plot_name in plot_names:
            if self.__explanation_type == 'global':
                plot = self.__global_plot_options[plot_name](self.__model, self.__explanation_type,
                                                             self.__instance_loc)
            else:
                plot = self.__local_plot_options[plot_name](self.__model, self.__explanation_type,
                                                            self.__instance_loc)
            plot.generate_explanation(explanations)

    def _validate_plot_names(self, plot_names: list[str]):
        for plot in plot_names:
            if plot not in self.__local_plot_options and plot not in self.__global_plot_options:
                raise ValueError(f"The given plot name \"{plot}\" is not valid.")

    def generate_explanation_visualizations(self, plot_names: list[str] = None):
        """Public method that generates the desired explanations visualizations."""
        if plot_names is None:
            if self.__explanation_type == 'local':
                plot_names = self.__local_plot_options.keys()
            elif self.__explanation_type == 'global':
                plot_names = self.__global_plot_options.keys()
        self._validate_plot_names(plot_names)
        explanations = self._generate_explanation(self.__features)
        self._generate_explanation_plots(explanations, plot_names)
