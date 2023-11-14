import shap
from sklearn.base import is_classifier
from . import Plot

class SHAPWaterfallPlot(Plot):
    """
    Class that generates waterfall plots for SHAP local explanations and also gives a verbal explanation
    to help on the visualization's interpretation.
    """
    objective: str = "Waterfall plot explanation"

    def generate_explanation(self, explanations):
        print(self.objective)
        if self._explanation_type == 'local':
            if is_classifier(self._model):
                shap.plots.waterfall(explanations[:, :, 1])
            else:
                shap.plots.waterfall(explanations[:, :])
        else:
            raise ValueError('The waterfall plot only supports local explanations.')
