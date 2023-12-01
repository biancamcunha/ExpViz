import numpy as np
from pandas import DataFrame
import shap
from sklearn.base import is_classifier
from .enums import VisualizationsObjectivesEnum
from . import Plot

class SHAPBarPlot(Plot):
    """
    Class that generates bar plots for SHAP global or local explanations and also gives a textual
    explanation to help on the visualization's interpretation.
    """
    objective: str = VisualizationsObjectivesEnum.SHAP_BAR_PLOT.value

    @staticmethod
    def __get_global_textual_explanation(explanations) -> None:
        df_bar_plot = DataFrame()
        df_bar_plot['feature_name'] = explanations[0].feature_names
        df_bar_plot['mean_shap_value'] = np.abs(explanations.values).mean(0)[:,0]
        df_bar_plot.sort_values(by='mean_shap_value', ascending=False, ignore_index=True,
                                inplace = True)
        l_feature_names = list(df_bar_plot['feature_name'].values)
        text_exp = f'''This plot shows the features sorted by magnitude of impact on the model in
        general, considering their absolute mean SHAP values.  In this particular case, the 5
        features that presented highest impact in the model were {l_feature_names[0]},
        {l_feature_names[1]}, {l_feature_names[2]}, {l_feature_names[3]} and
        {l_feature_names[4]}.'''
        print(text_exp)

    def __get_local_textual_explanation(self, explanations: any) -> None:
        df_bar_plot = DataFrame()
        df_bar_plot['feature_name'] = explanations[0].feature_names
        df_bar_plot['shap_value'] = explanations[0].values[:,0]
        df_bar_plot.sort_values(by='shap_value', ascending=False, ignore_index=True, inplace = True)
        l_feature_names = list(df_bar_plot['feature_name'].values)
        if is_classifier(self._model):
            text_exp = f'''This plot shows the features sorted by magnitude of impact on the model
            considering their absolute SHAP values. It shows the magnitude of the impact as well as
            if the contribution was towards a class or another. The bars in red represent the SHAP
            values of features that contributed for the positive class, and therefore grow to the
            right side of the plot and have their value displayed with a positive sign. The bars is
            blue represent the SHAP values of features that contributed for the negative class, and
            therefore grow to the left side of the plot and have their value displayed with a
            negative sign. For the chosen observation, {l_feature_names[0]}, {l_feature_names[1]}
            and {l_feature_names[2]} had the highest contribution for the positive class and
            {l_feature_names[-1]}, {l_feature_names[-2]}, {l_feature_names[-3]} had the highest
            contribution for the negative class.'''
        else:
            text_exp = f'''This plot shows the features sorted by magnitude of impact on the model
            considering their absolute SHAP values. It shows the magnitude of the impact as well as
            if the contribution was positive or negative. The bars in red represent the SHAP values
            of features that had a positive contribution, and therefore grow to the right side of
            the plot and have their value displayed with a positive sign. The bars in blue
            represent the SHAP values of features that contributed for the negative class, and
            therefore grow to the left side of the plot and have their value displayed with a
            negative sign. For the chosen observation, {l_feature_names[0]}, {l_feature_names[1]}
            and {l_feature_names[2]} had the highest positive contribution and
            {l_feature_names[-1]}, {l_feature_names[-2]}, {l_feature_names[-3]} had the highest
            negative contribution.'''
        print(text_exp)

    def _display_textual_explanation(self, explanations: any) -> None:
        if self._explanation_type == 'global':
            self.__get_global_textual_explanation(explanations)
        else:
            self.__get_local_textual_explanation(explanations)


    def generate_explanation(self, explanations: any) -> None:
        self._display_textual_explanation(explanations)
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
