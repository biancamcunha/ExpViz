import shap
from sklearn.base import is_classifier
from . import Plot
from . import VisualizationsObjectivesEnum

class SHAPWaterfallPlot(Plot):
    """
    Class that generates waterfall plots for SHAP local explanations and also gives a verbal explanation
    to help on the visualization's interpretation.
    """
    objective: str = VisualizationsObjectivesEnum.SHAP_WATERFALL_PLOT.value

    @staticmethod
    def _get_textual_explanation():
        pass

    def generate_explanation(self, explanations: any):
        self._get_textual_explanation()
        if self._explanation_type == 'local':
            if is_classifier(self._model):
                shap.plots.waterfall(explanations[:, :, 1])
            else:
                shap.plots.waterfall(explanations[:, :])
        else:
            raise ValueError('The waterfall plot only supports local explanations.')
