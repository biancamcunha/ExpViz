from . import Plot

class LIMEPlot(Plot):
    """
    Class that generates visualization for LIME explanations and also gives a verbal explanation
    to help on the visualization's interpretation.
    """
    objective: str = "LIME plot explanation"

    def generate_explanation(self, explanations):
        print(self.objective)
        explanations.show_in_notebook(show_table=True)
