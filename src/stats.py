import pandas as pd
import numpy as np
import scipy.stats

def chi2_stats(df: pd.DataFrame) -> pd.DataFrame :

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
    
    df_factors.sort_values(by='chi', ascending=False, inplace=True)

    return df_factors


def binom_stats(df, df_factors):

    P_DISSATISFACTION = df['user_dissatisfied'].mean() # Overall probability of dissatisfied response

    df_satisfaction = pd.DataFrame()

    for fct in df_factors.loc[(df_factors['variable_type']=="analyse"),'factor']:
        ct_cluster_satisfaction = pd.crosstab(df[fct],df['user_dissatisfied'])
        ct_cluster_satisfaction.columns=['satisfied_count','dissatisfied_count']
        ct_cluster_satisfaction.index.name='value'
        ct_cluster_satisfaction['factor']=fct
        df_satisfaction = pd.concat([df_satisfaction,ct_cluster_satisfaction])


    df_satisfaction = df_satisfaction.reset_index()
    df_satisfaction['total']=df_satisfaction['satisfied_count']+df_satisfaction['dissatisfied_count']
    df_satisfaction['dissatisfied_ratio'] = df_satisfaction['dissatisfied_count']/df_satisfaction['total']
    df_satisfaction['probability'] = df_satisfaction.apply(lambda x: scipy.stats.binom.cdf(k=x['dissatisfied_count'],n=x['total'],p=P_DISSATISFACTION ), axis=1)
    
    df_satisfaction = pd.merge(df_satisfaction, df_factors, on='factor', how='left')
    df_satisfaction.sort_values(by=['chi','factor','dissatisfied_ratio'],ascending=[False,False,False],inplace=True)

    return(df_satisfaction[['factor','value','dissatisfied_ratio','satisfied_count','dissatisfied_count','total','probability','chi']])

