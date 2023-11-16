from . import Plot
from . import VisualizationsObjectivesEnum

class LIMEPlot(Plot):
    """
    Class that generates visualization for LIME explanations and also gives a verbal explanation
    to help on the visualization's interpretation.
    """
    objective: str = VisualizationsObjectivesEnum.LIME_PLOT.value

    @staticmethod
    def _get_textual_explanation():
        pass

    def generate_explanation(self, explanations: any):
        self._get_textual_explanation()
        explanations.show_in_notebook(show_table=True)
