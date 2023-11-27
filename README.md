# ExpViz
ExpViz is a library that will provide visualizations and verbal explanations to the prediction explanations generated by widely used Machine Learning model explanation methods, such as SHAP and LIME. It is a wrapper library with the objective of having the most used explanation methods all in one place, and also having their explanations and visualizations generated using the same sintax and making the process of comparing the explanations and choosing the best one easier. It is also useful for researchers who, like me, happen to be studying ML model explanations and explanation visualizations and need to be able to analyze different explanation methods and visualizations more efficiently. Additionally, along with the visualizations, we provide a textual explanation of the exhibited visualization to help the user to interpret the visualization and better understand the explanation. 

The first version of ExpViz supports the SHAP and LIME methods, since these two are some of the most popular methods and have been widely used in the XAI field. Future versions will possibly cover other XAI methods. 

## Motivation
Explainable AI has been a growing field over the past years and several methods have been developed with the objective to improve Machine Learning models' interpretability by providing explanations for the models' behavior or for specific instance predictions. However, these novel methods have been proposed with the claim that their explanations are interpretable, without actually proving that claim. Having that in mind, the motivation of ExpViz was first to facilitate the work of researchers that are studying how of the existing explanations can or can not help in improving interpretability, and also to try to take another step towards better interpretability by giving a verbal explanation to these visualizations.

## Installation
1. Clone this repository using the command
```
git clone git@github.com:biancamcunha/ProjetoFinalDeProgramacao.git
```
2. Go to the root directory of the project and use the command
```
python setup.py install
```

## Brief overview of the XAI methods covered by ExpViz
### SHAP
**SHAP (SHapley Additive exPlanations)** is an approach to explain the output of any machine learning model that leverages the cooperative game theory concept of Shapley values. It is a method used in order to increase transparency and interpretability of ML models, specially for complex black-box models, such as neural networks. The method was proposed by Lundberg and Lee in 2017, and has been largely used in the following years for purposes of explaining behavior of ML models or specific predictions. The core idea behind explanations for ML models based on Shapley values is to use fair allocation of results from cooperative game theory to allocate credit for a model's output among its input features. 

One property of Shapley values is that the Shapley values of all the input features will always sum up to the difference between the expected model output _**E[f(x)]**_ and the actual model output of the prediction being explained _**f(x)**_. That can be easily seen throug one of the visualizations present in the SHAP library, called waterfall plot, that is also available in ExpViz:

![waterfallplot](.\images\waterfall_plot.png)

All the visualizations that are available in ExpViz and how they work will be addressed in the Visualizations section.

### LIME
**LIME (Local Interpretable Model-agnostic Explanations)** generates explanations for predictions generated by any machine learning classifier by learning an interpretable model locally around the prediction instance. It was proposed by Ribeiro et al. in 2016 and also presented SP-LIME, a method for selecting representative and non-redundant explanations in order to generate a global explanation for the model. However, the LIME python library only supports local explanations for text, tabular or image classifiers.

## Architecture
ExpViz is composed by the packages _exp_explainers_ and _visualizations_. For the exp_explainers, ther is an interface called _Explainer_ from which all the encompassed explanation methods will inherit and implement the base functions. Each of them will have a set of visualizations that will be available for use. The visualizations com from the other package and also have an interface called _Plot_. Each of the possible visualizations will display their plot and also have a parameterized verbal explanation, that helps the user to interpret the visualization considering the data used by them.
This structure makes the solution more scalable, since it is easy to add new explanation methods and new visualizations without having to alter the ones that already exist. 

## Visualizations

### SHAP

In order to illustrate the SHAP visualizations, we will use the [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset). The target variable of this dataset is the median house value for California districts, expressed in hundreds of thousands of dollars.

#### Bar plot
The SHAP library can generate global and local bar plots. The **global** bar plot shows the mean of the absolute SHAP values for each feature sorted from the feature with the highest mean to the feature with the lowest mean. That is, the plot will visually show the features sorted by how much impact it has on the predictions, displaying the impact's magnitude through the bars. The y-axis has the features in order of impact and the x-axis has the mean of the absolute SHAP values. It is a very usefull visualization to understand which features influence most in the predictions, however it gives no indication if the impact is positive or negative, since the mean is calculated using the absolute SHAP values.

![globalbarplot](.\images\global_bar_plot.png)

The **local** bar plot on the other hand shows the SHAP values of the features for a specific prediction, also sorting the features by magnitude of the impact. It explicits not only the magnitude but also the direction of the impact (positive or negative). 

![localbarplot](.\images\local_bar_plot.png)

#### Waterfall plot

The waterfall plot supports only local explanations. In this plot, the x-axis has the possible values for the target variable, in this case the median house value, instead of the SHAP values. The plot is constructed originating in the expected value of the target variable **E[f(x)]**, which is the mean of the predictions. Starting from the expected value, a bar with the SHAP value of the feature with the least impact is inserted and a new starting point is set for the next feature's bar. If the SHAP value is positive, the bar will grow to the right side of the graph, otherwise it will grow to the left side. After intserting the bars for every feature, the y-axis is complete will all the features, sorted by the impact in the prediction, and the last bar ends in the predicted value. This plot shows how each feature moved the predicted value away from the expected value and made the model get to that prediction.

![waterfallplot](.\images\waterfall_plot.png)

#### Beeswarm
The beeeswarm plot is a visualization for global explanations. Like the global bar plot, it shows the features sorted by the impact in the model's output. However, it offers more information than the previous visualization. IT has a vertical bar on the right side of the plot that displays a color range of the feature value from high to low. The distribution of the features' SHAP values is plotted along the x-axis and the dots are colored according to the before mentioned color range. Looking at the examples below, we can see that MedInc is the feature that has the highest impact in the prediction, just like we observed in the bar plot, but we also know that the low values of the feature cause a negative impact on the prediction and high values cause a positive impact. Latitude on the other hand has the inverse effect, i.e., the higher the latitude the more negative is the impact on the output.

![beeswarmplot](.\images\beeswarm_plot.png)


### LIME

For the LIME visualization example we use the [Wine Quality Dataset](https://www.kaggle.com/datasets/piyushagni5/white-wine-quality), which has information about wines' characteristics and the target variable is _quality_, that can be of classes good or bad.

The LIME library only has one type of visualization for its explanations. It is divided in three parts. The first one, to the left, shows the target classes of the problem and the probability generated by the model for each of them. The one in the middle shows the most important features, ordenated from most important to least important, showing how much each of them contributed to the final probabilities and for which class they contributed. As explained before, LIME explanations come from more interpretable models that are trained locally around the predicted instance, and the plot in the middle also gives the conditions from the trained model that were activated by the feature value along with their contribution. The third part of the visualization gives the list of features, also sorted by importance, and their values in the predicted instance and also colors each line with the color that represents the class that they contributed to.

![limeplot](.\images\lime_plot.png)

## Quick start

In this section we will show how to use the ExpViz library to generate explanations for your models using a desired explanation methods. 

1. Import the library and other libraries you will need to get the data and train the model.
```
import exp_viz
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
```

2. Get the data and train the model.
```

# California Housing Prices
dataset = fetch_california_housing(as_frame = True)
X = dataset['data']
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Prepares a default instance of the random forest regressor
model = RandomForestRegressor()

# Fits the model on the data
model.fit(X_train, y_train)
```

3. Choose the explainer you want to use, in this case we will use the SHAPExplainer, and instantitate it.
```
shap_exp = exp_viz.SHAPExplainer(model, X_test, 'global')
```

4. If you need to know which visualizations are available for each explainer, use the class function _list_visualization_options()_.
```
exp_viz.SHAPExplainer.list_visualization_options()
```

6. If you want to know the objective of each available visualization, use the class function _get_visualization_objective()_ giving the name of the desired plot as parameter.
```
exp_viz.SHAPExplainer.get_visualization_objective("waterfall_plot")
```

7. Use the method _generate_explanation_visualizations()_ method in order to generate and display the visualizations nad textual explanations.
```
shap_exp.generate_explanation_visualizations()
```
