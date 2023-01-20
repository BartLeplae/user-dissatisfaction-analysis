""" stats:
    Apply statistics on the incident tickets and store results in factors dataframe
Input:
    - dataframe with incident tickets
    - dataframe with factors 
Output:
    - factor dataframe
    - factor-values dataframe
"""

import pandas as pd
import numpy as np
import scipy.stats

def chi2_stats(df: pd.DataFrame) -> pd.DataFrame :
    """ apply chi2 statistic on the different columns of the incident tickets
    Input: dataframe with incident tickets
    Returns: new factors database
    """
    # analyse the df_factors: dataframe with all available columns
    df_factors = pd.DataFrame(df.columns.values.tolist())
    df_factors.columns=(['factor'])
    df_factors['variable_type']="analyse"  # attribute to store the action: analyze, 

    # identify the data type for every of the df_factors
    for fct in df_factors['factor']:
        df_factors.loc[df_factors['factor']==fct,'dtype'] = df[fct].dtype

    # identify the number of unique values for every of the factdors
    for fct in df_factors['factor']:
        df_factors.loc[df_factors['factor']==fct,'unique_values'] = df[fct].nunique()

    df_factors.loc[df_factors['factor']=="user_dissatisfied",'variable_type'] = 'response'

    # for every of the factors: calculate the chi2 and p scores
    # to determine if the factor values are a differentiator
    for fct in df_factors.loc[df_factors['variable_type']=="analyse",'factor']:
        ct_cluster_satisfaction = pd.crosstab(df[fct],df['user_dissatisfied'])
        chi, p, dof, expected  = scipy.stats.chi2_contingency(ct_cluster_satisfaction)
        df_factors.loc[df_factors['factor']==fct,'chi'] = chi
        df_factors.loc[df_factors['factor']==fct,'p'] = p
    
    # sort the factors so that the most differentiating factors are listed first
    df_factors.sort_values(by='chi', ascending=False, inplace=True)

    return df_factors


def binom_stats(df, df_factors):
    """ apply cumulative binomial statistic on the distinct columns values of the incident tickets
        count the number of tickets for satisfied and dissatisfied responses
        determine the ratio of dissatisfied responses
    Input: dataframe with incident tickets, dataframe with the factors
    Returns: dataframe with factor - value combinations
    """

    df_factor_values = pd.DataFrame()

    # for every factor - value combination, determine the satisfied dissatisfied counts and add to df_satisfaction
    for fct in df_factors.loc[(df_factors['variable_type']=="analyse"),'factor']:
        ct_cluster_satisfaction = pd.crosstab(df[fct],df['user_dissatisfied'])
        ct_cluster_satisfaction.columns=['satisfied_count','dissatisfied_count']
        ct_cluster_satisfaction.index.name='value'
        ct_cluster_satisfaction['factor']=fct
        df_factor_values = pd.concat([df_factor_values,ct_cluster_satisfaction])

    # for every factor - value combination, calculate the total tickets and the ratio of dissatisfied responses
    df_factor_values = df_factor_values.reset_index()
    df_factor_values['total']=df_factor_values['satisfied_count']+df_factor_values['dissatisfied_count']
    df_factor_values['dissatisfied_ratio'] = df_factor_values['dissatisfied_count']/df_factor_values['total']

    return(df_factor_values)
    


