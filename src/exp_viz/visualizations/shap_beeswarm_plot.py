import shap
from sklearn.base import is_classifier
from . import Plot

class SHAPBeeswarmPlot(Plot):
    """
    Class that generates beeswarm plots for SHAP global explanations and also gives a verbal explanation
    to help on the visualization's interpretation.
    """
    objective: str = "Beeswarm plot explanation"

    def generate_explanation(self, explanations):
        print(self.objective)
        if self._explanation_type == 'global':
            if is_classifier(self._model):
                shap.plots.beeswarm(explanations[:, :, 1])
            else:
                shap.plots.beeswarm(explanations[:, :])
        else:
            raise ValueError('The beeswarm plot only supports global explanations.')
