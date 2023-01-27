""" model:
    model the ratio of dissatisfied responses with the available factors
    this model identifies the most important causal factors
Input:
    - X and y
Output:
    - regression model
"""
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import graphviz 
from sklearn.tree import export_text
# import seaborn as sns

def DecisionTree(X,y):
    """ build decision tree that predicts user dissatisfaction ratios based on causal factors
    Use a score that evaluate the correctness of the predicted dissatisfaction % across the entire range of satisfaction scores
    Do this instead of trying to correctly predict the satisfaction response for individual tickets
    Input: 
        X: available contribution factors  
        y: actual user dissatisfaction responses
    Returns: the model
    """
    
    def score_func(y_true, y_pred):
        """ custom score function to select the hyperparameters that provide the least differences across
            the full range of actual dissatisfaction ratios 
            the performance is determined by testing against 10 percentile ranges
        Input: 
            y_true: the actual user dissatisfaction
            y_pred: predicted user dissatisfaction
        Returns: a score which is higher for better fits
        """
        d = {'actual': y_true, 'prob': y_pred}

        # split the data in 10 groups based on the dissatisfaction prediction score
        # for each of these groups: test to what extent the predicted score for the groups is different from the actual dissatisfaction%
        df_test = pd.DataFrame(data=d)
        df_test['dissatisfaction_rank'] = df_test['prob'].rank(pct=True, method='dense')
        df_test['dissatisfaction_rank'] = df_test['dissatisfaction_rank']*10
        df_test['dissatisfaction_rank'] = df_test['dissatisfaction_rank'].round(0)
        df_test['dissatisfaction_rank'] = df_test['dissatisfaction_rank'].astype('int')

        pivot_test = pd.pivot_table(df_test,index=['dissatisfaction_rank',],values=['actual','prob'],aggfunc='sum')
        pivot_test['diff']=(pivot_test['prob']-pivot_test['actual']).abs()

        return (1/(pivot_test['diff'].mean()))  # 1/x  to return a higher score when the sum of the absolue differences is lower

    # we are looking to match the probability across a range of tickets, rather than seeking to predict user dissatisfaction on a per ticket basis
    score = make_scorer(score_func, greater_is_better=True, needs_proba=True)  

    clf = tree.DecisionTreeClassifier()

    # Hyperparameters for GridSearchCV
    params = {
        'max_depth': [5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [50, 60, 70, 80, 90, 100, 110, 120, 130],
        'criterion': ["gini","entropy"]
    }

    grid_search = GridSearchCV( estimator=clf, 
                                param_grid=params, 
                                n_jobs=-1, verbose=1, cv=5, scoring = score)

    grid_search.fit(X, y)

    # Store the result for each of the Hyperparameter combinations
    # score_df = pd.DataFrame(grid_search.cv_results_)
    # score_df.to_excel('score.xlsx')

    # Retrun the best model
    clf_best = grid_search.best_estimator_

    return clf_best
