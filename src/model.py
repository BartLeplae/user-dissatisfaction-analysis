""" model:
    model the ratio of dissatisfied responses
Input:
    - X and y
Output:
    - regression model
"""
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
# import seaborn as sns

def DecisionTree(X,y):

    def score_func(y_true, y_pred):
        d = {'actual': y_true, 'prob': y_pred}
        df_test = pd.DataFrame(data=d)
        df_test['dissatisfaction_rank'] = df_test['prob'].rank(pct=True, method='dense')
        df_test['dissatisfaction_rank'] = df_test['dissatisfaction_rank']*10
        df_test['dissatisfaction_rank'] = df_test['dissatisfaction_rank'].round(0)
        df_test['dissatisfaction_rank'] = df_test['dissatisfaction_rank'].astype('int')

        pivot_test = pd.pivot_table(df_test,index=['dissatisfaction_rank',],values=['actual','prob'],aggfunc='sum')
        pivot_test['diff']=(pivot_test['prob']-pivot_test['actual']).abs()
        # print(pd.pivot_table(df_test,index=['dissatisfaction_rank',],values=['prob','actual'],aggfunc=['mean','count']))
        return (1/(pivot_test['diff'].mean()))

    loss = make_scorer(score_func, greater_is_better=False, needs_proba=True)
    score = make_scorer(score_func, greater_is_better=True, needs_proba=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.333, random_state=135)

    clf = tree.DecisionTreeClassifier()

    params = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'min_samples_split': [0.8,0.9,0.10,0.11,0.12, 0.15,],
        # 'min_samples_leaf': [5, 10, 15, 20, 25, 30, 50, 100],
        'criterion': ["gini", "entropy"]
    }
    # params = {
    #     'max_depth': [3,4],
    #     'min_samples_split': [0.10, 0.12],
    #     'min_samples_leaf': [5,6],
    #     'criterion': ["gini"]
    # }
    grid_search = GridSearchCV( estimator=clf, 
                                param_grid=params, 
                                n_jobs=-1, verbose=1, cv=6, scoring = score)
    # grid_search.fit(X_train, y_train)
    grid_search.fit(X, y)

    # y_pred = clf.predict(X_test)

    # print(precision_recall_fscore_support(y_test, y_pred, average='binary'))
    # print(clf.n_features_in_, clf.feature_importances_)
    # print(confusion_matrix(y_test,y_pred))

    score_df = pd.DataFrame(grid_search.cv_results_)
    score_df.to_excel('score.xlsx')

    clf_best = grid_search.best_estimator_
    print(clf_best)

    # def get_dt_graph(dt_classifier):
    #     fig = plt.figure(figsize=(25,20))
    #     _ = tree.plot_tree(dt_classifier,
    #                    feature_names=X.columns,
    #                    class_names=["Satisfied", "Dissatisfied"],
    #                    filled=True)

    # get_dt_graph(dt_best)

    return clf_best
