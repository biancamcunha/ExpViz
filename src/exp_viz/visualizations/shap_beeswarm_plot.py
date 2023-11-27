import numpy as np
from pandas import DataFrame
import shap
from sklearn.base import is_classifier
from . import Plot
from .enums import VisualizationsObjectivesEnum

class SHAPBeeswarmPlot(Plot):
    """
    Class that generates beeswarm plots for SHAP global explanations and also gives a textual explanation
    to help on the visualization's interpretation.
    """
    objective: str = "Beeswarm plot explanation"

    def _display_textual_explanation(self, explanations: any) -> None:
        df_bar_plot = DataFrame()
        df_bar_plot['feature_name'] = explanations[0].feature_names
        df_bar_plot['mean_shap_value'] = np.abs(explanations.values).mean(0)[:,0]
        df_bar_plot.sort_values(by='mean_shap_value', ascending=False, ignore_index=True,
                                inplace = True)
        l_feature_names = list(df_bar_plot['feature_name'].values)
        text_exp = f'''This plot shows the features sorted by magnitude of impact on the model in
        general, considering their absolute mean SHAP values. Each feature has a beeswarm and they
        are composed by dots that represent each observation and are distributed along the x-axis
        according to the observations' SHAP values. On the right side of the plot there is a
        vertical bar that gives the color shades of the dots that represent from high to low values
        of the features. Red dots represent high feature values, shades of purple represent medium
        values and blue dots represent low values. That means that along with the information of
        how much impact each feature has on the model output, it is possible to know what ranges
        of feature values have positive, negative or no impact. In this particular case, the 5
        features that presented highest impact in the model were {l_feature_names[0]},
        {l_feature_names[1]}, {l_feature_names[2]}, {l_feature_names[3]} and {l_feature_names[4]}.
        '''
        print(text_exp)

    def generate_explanation(self, explanations: any) -> None:
        self._display_textual_explanation(explanations)
        if self._explanation_type == 'global':
            if is_classifier(self._model):
                shap.plots.beeswarm(explanations[:, :, 1])
            else:
                shap.plots.beeswarm(explanations[:, :])
        else:
            raise ValueError('The beeswarm plot only supports global explanations.')
