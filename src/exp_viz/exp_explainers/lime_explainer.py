#pylint: disable=arguments-differ
from pandas import DataFrame
from lime import lime_tabular
from sklearn.base import is_classifier
from .explainer import Explainer
from ..visualizations import LIMEPlot

class LIMEExplainer(Explainer):
    """
    Class that generates explanations for Machine Learning models by using LIME explanation
    method.
    """

    def __init__(self, model, features, instance_loc) -> None:
        self.__explainer: object = None
        self.__model: object = model
        self.__features: DataFrame = features
        self.__instance_loc: int = instance_loc

    def _generate_explanation(self, features: DataFrame) -> any:
        if is_classifier(self.__model):
            mode = 'classification'
        else:
            mode = 'regression'
        self.__explainer = lime_tabular.LimeTabularExplainer(training_data=features.values,
                                                           feature_names=features.columns.values,
                                                           discretize_continuous=True,
                                                           mode=mode,
                                                           verbose=True,
                                                           random_state=123)
        return self.__explainer.explain_instance(features.values[self.__instance_loc,:],
                                               self.__model.predict_proba,
                                               num_features=10)

    def _generate_explanation_plots(self, explanations: any) -> None:
        plot = LIMEPlot()
        plot.generate_explanation(explanations)

    def generate_explanation_visualizations(self) -> None:
        """Public method that generates the desired explanations visualizations."""
        explanation = self._generate_explanation(self.__features)
        self._generate_explanation_plots(explanation)

    @classmethod
    def list_visualization_options(cls) -> None:
        print("Visualization options:")
        print(list('lime_plot'))

    @classmethod
    def get_visualization_objective(cls, plot_name: str) -> None:
        print(LIMEPlot.objective)
