import shap
from sklearn.base import is_classifier
from . import Plot
from . import VisualizationsObjectivesEnum

class SHAPBarPlot(Plot):
    """
    Class that generates bar plots for SHAP global or local explanations and also gives a verbal
    explanation to help on the visualization's interpretation.
    """
    objective: str = VisualizationsObjectivesEnum.SHAP_BAR_PLOT.value

    @staticmethod
    def _get_textual_explanation():
        pass

    def generate_explanation(self, explanations: any):
        self._get_textual_explanation()
        if is_classifier(self._model):
            if self._explanation_type == 'global':
                shap.plots.bar(explanations[:, :, 1])
            else:
                shap.plots.bar(explanations[self._instance_loc, :, 1])
        else:
            if self._explanation_type == 'global':
                shap.plots.bar(explanations)
            else:
                shap.plots.bar(explanations[self._instance_loc])
