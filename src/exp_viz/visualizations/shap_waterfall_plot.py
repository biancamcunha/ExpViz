from pandas import DataFrame
import shap
from sklearn.base import is_classifier
from . import Plot
from . import VisualizationsObjectivesEnum

class SHAPWaterfallPlot(Plot):
    """
    Class that generates waterfall plots for SHAP local explanations and also gives a textual explanation
    to help on the visualization's interpretation.
    """
    objective: str = VisualizationsObjectivesEnum.SHAP_WATERFALL_PLOT.value

    def _display_textual_explanation(self, explanations: any) -> None:
        df_bar_plot = DataFrame()
        df_bar_plot['feature_name'] = explanations[0].feature_names
        df_bar_plot['shap_value'] = explanations[0].values[:,0]
        df_bar_plot.sort_values(by='shap_value', ascending=False, ignore_index=True, inplace = True)
        l_feature_names = list(df_bar_plot['feature_name'].values)
        if is_classifier(self._model):
            text_exp = f'''In this plot, the x-axis has the possible values for the target variable
            instead of the SHAP values, and the y-axis has the feature names. The features are
            sorted by magnitude of impact on the model considering their absolute SHAP values.
            In the x-axis there is also the representation of the target expected value E[f(X)],
            which is the mean target value of all the predictions. It shows how much each feature
            contributed for that prediction to have been higher or lower than the expected value.
            The bars in red represent the SHAP values of features that contributed for the positive
            class, and therefore grow to the right side of the plot. The bars in blue represent the
            SHAP values of features that contributed for the negative class, and therefore grow to
            the left side of the plot.  For the chosen observation, {l_feature_names[0]},
            {l_feature_names[1]} and {l_feature_names[2]} had the highest contribution for the
            positive class and {l_feature_names[-1]}, {l_feature_names[-2]}, {l_feature_names[-3]}
            had the highest contribution for the negative class.'''
        else:
            text_exp = f'''In this plot, the x-axis has the possible values for the target variable
            instead of the SHAP values, and the y-axis has the feature names. The features are
            sorted by magnitude of impact on the model considering their absolute SHAP values.
            In the x-axis there is also the representation of the target expected value E[f(X)],
            which is the mean target value of all the predictions. It shows how much each feature
            contributed for that prediction to have been higher or lower than the expected value.
            The bars in red represent the SHAP values of features that had a positive contribution,
            and therefore grow to the right side of the plot. The bars in blue represent the
            SHAP values of features that had a positive contribution, and therefore grow to the
            left side of the plot. For the chosen observation, {l_feature_names[0]},
            {l_feature_names[1]} and {l_feature_names[2]} had the highest positive contribution and
             {l_feature_names[-1]}, {l_feature_names[-2]}, {l_feature_names[-3]}
            had the highest negative contribution.'''
        print(text_exp)

    def generate_explanation(self, explanations: any) -> None:
        self._display_textual_explanation(explanations)
        if self._explanation_type == 'local':
            if is_classifier(self._model):
                shap.plots.waterfall(explanations[self._instance_loc, :, 1])
            else:
                shap.plots.waterfall(explanations[self._instance_loc, :])
        else:
            raise ValueError('The waterfall plot only supports local explanations.')
