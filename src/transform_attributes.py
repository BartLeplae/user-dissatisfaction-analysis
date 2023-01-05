""" transform_attributes:
    Modifies / transforms incident ticket dataframes at different stages of the review process
Input:
    - dataframe with incident tickets
    - dataframe with factors (list columns of interest)    
Output:
    - modified incident ticket and factor dataframes
"""
import pandas as pd
import numpy as np
from stats import chi2_stats, binom_stats

def transform_df_upon_db_retrieval (df):
    """ transform dataframe as retrieved from the database
    Input: dataframe with incident tickets
    Returns: modified dataframe
    """
    # Determinae the log2 of the number of times a knowledge article has been utilized
    df['ka_count'].fillna(0,inplace=True)
    df['ka_count_log'] = np.ceil(np.log2(df['ka_count']+1)/2)
    df.drop(columns='ka_count', inplace=True)

    # Transform time to resolve the incidents to logaritmic scale (based on number of days)
    df['ttr_days_log']=np.round(np.log2(1+df['am_ttr']/(24*3600))).astype('int')
    
    # Transorm time to resolve from seconds to days
    df['ttr_days']=np.round(1+df['am_ttr']/(24*3600)).astype('int')
    df.loc[df['ttr_days']>20,'ttr_days']=20
    df.drop(columns='am_ttr', inplace=True) # drop time to resolve in seconds
    
    # user is dissatisfied when survey response value is 1 or 2
    # code dissatisfied as 1
    # drop the survey_response_value column
    df['user_dissatisfied']=0
    df.loc[df['survey_response_value']<3,'user_dissatisfied']=1
    df = df.drop(columns='survey_response_value')

    # sla_breached = 1 when sla_result == 'Breached'
    df['sla_breached']=0
    df.loc[df['sla_result']=="Breached",'sla_breached']=1
    df = df.drop(columns='sla_result')

    # caller_is_employee = 1 when caller_employee_type == 'employees'
    df['caller_is_employee']=0
    df.loc[df['caller_employee_type']=="employees",'caller_is_employee']=1
    df = df.drop(columns='caller_employee_type')

    # priority_4 = 1 when sla_priority == 'Priority 4', all other priorities all bundled (VIP, Priority 2 and Priority 3)
    df['priority_is_4']=0
    df.loc[df['sla_priority']=="Priority 4",'priority_is_4']=1
    df = df.drop(columns='sla_priority')

    return df

def transform_df_upon_chi2 (df_factors):
    """ transform dataframe upon review of chi2 values
    Input: dataframe with the factors (columns of interest)
    Returns: modified factors dataframe
    """
    # Ignore the factors for which the p value is greater than 5%
    df_factors.loc[(df_factors['variable_type']=="analyse")&(df_factors['p']>0.05),"variable_type"] = "ignore"

    # Perform secondary analysis when number of values > 20 (application, resolving group, ...)
    df_factors.loc[(df_factors['variable_type']=="analyse")&(df_factors['unique_values']>20),"variable_type"] = "analyse2"  

    return (df_factors)

def transform_df_upon_review_values (df_incidents, df_factors):
    """ transform incident and factors dataframes upon review of the individual values
    Input: dataframes with the incidents, dataframe with factors (columns of interest)
    Returns: modified dataframes
    """
    # Considering the similarities in dissatisfaction ratio and limited total, limit ttr_days_log to 5 
    df_incidents.loc[df_incidents['ttr_days_log']>5,'ttr_days_log']=5

    # Ignore the ttr_days for subsequent analysis
    df_factors.loc[(df_factors['factor']=="ttr_days"),"variable_type"] = "ignore"

    # Limit reassignment count to given the low values for higher reassignment counts
    df_incidents.loc[df_incidents['reassignment_count']>4,'reassignment_count']=4

    # Reclassify close codes with less than 150 tickets to 'Environmental Restoration'
    df_incidents.loc[df_incidents['close_code'].isin(
        ['Capacity Adjustment','Hardware Correction','Redundancy Activation']),'close_code']="Environmental Restoration"

    # plan assignment_group_company secondary analysis
    df_factors.loc[(df_factors['factor']=="assignment_group_company"),"variable_type"] = "analyse2"

    # Ignore the ka_count_log for subsequent analysis given the lack of correlation with dissatisfaction
    df_factors.loc[(df_factors['factor']=="ka_count_log"),"variable_type"] = "ignore"

    # Ignore the contact_type for subsequent analysis since 'self_service' is a better differentiator
    df_factors.loc[(df_factors['factor']=="contact_type"),"variable_type"] = "ignore"

    # Ignore the breached_reason_code for subsequent analysis since values are insufficiently differentiated or have low occurences
    df_factors.loc[(df_factors['factor']=="breached_reason_code"),"variable_type"] = "ignore"

    # Ignore the appl_tier for subsequent analysis since values are insufficiently differentiated or have low occurences
    df_factors.loc[(df_factors['factor']=="appl_tier"),"variable_type"] = "ignore"

    return (df_incidents, df_factors)


def df_create_dummies (df_incidents, df_factors):
    """ create dummy columns in df_incidents
    Input: dataframe with the incidents, dataframe with the factors (columns of interest)
    Returns: modified df_incident and df_factors dataframes
    """
    df_incidents = df_incidents.join(pd.get_dummies(df_incidents['close_code'], prefix='close_code'))
    df_factors.loc[(df_factors['factor']=="close_code"),"variable_type"] = "one_hot_encoded"
    df_factors = pd.concat([df_factors, chi2_stats(df_incidents)])
    df_factors.drop_duplicates(subset='factor', keep='first', inplace=True)
    df_factors.sort_values(by=['chi'],ascending=False,inplace=True)
    return (df_incidents, df_factors)

def create_Xy (df_incidents, df_factors, max_factors):
    """ subset the columns of df_incidents to those identified as 'analyse' in factors
    Input: dataframe with the incidents, dataframe with the factors, maximum number of factors to consider
    Returns: modified df_incident and df_factors dataframes
    """
    X_columns = df_factors[df_factors['variable_type']=='analyse'].factor
    X_columns = X_columns[:max_factors]

    X = df_incidents[X_columns]
    y_column = df_factors[df_factors['variable_type']=='response'].factor
    y = df_incidents[y_column]
    return X, y