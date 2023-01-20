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
from unique_names_generator import get_random_name
from unique_names_generator.data import NAMES, STAR_WARS, ANIMALS
import random as random

def transform_df_upon_db_retrieval (df):
    """ transform dataframe as retrieved from the database
    Input: dataframe with incident tickets
    Returns: modified dataframe
    """
    
    # Transorm time to resolve from seconds to days and truncate to 15 days
    df['days_to_resolve']=np.round(df['am_ttr']/(24*3600)).astype('int')
    df.loc[df['days_to_resolve']>15,'days_to_resolve']=15 
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

    # anonymise company
    df["assignment_group_company"].fillna("None",inplace=True)   
    companies = df["assignment_group_company"].unique()
    df_companies = pd.DataFrame(data={"assignment_group_company":companies,"company":""})
    df_companies["company"] = df_companies.apply(lambda x: "C"+str(random.randint(10000,99999)), axis=1)
    df = pd.merge(df,df_companies, on="assignment_group_company")
    df.drop(columns=["assignment_group_company"],inplace=True)

    # anonymise assignment group
    df["assignment_group_name"].fillna("None",inplace=True)
    groups = df["assignment_group_name"].unique()
    df_groups = pd.DataFrame(data={"assignment_group_name":groups,"group":""})
    df_groups["group"] = df_groups.apply(lambda x: "G"+str(random.randint(10000,99999)), axis=1)
    df = pd.merge(df,df_groups, on="assignment_group_name")
    df.drop(columns=["assignment_group_name"],inplace=True)

    # anonymise application name
    df["ci_name"].fillna("None",inplace=True)
    applications = df["ci_name"].unique()
    df_applications = pd.DataFrame(data={"ci_name":applications,"application":""})
    df_applications["application"] = df_applications.apply(lambda x: "A"+str(random.randint(10000,99999)), axis=1)
    df = pd.merge(df,df_applications, on="ci_name")
    df.drop(columns=["ci_name"],inplace=True)
    
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

    # Limit reassignment count to given the low values for higher reassignment counts
    df_incidents.loc[df_incidents['reassignment_count']>4,'reassignment_count']=4

    # Reclassify close codes with less than 150 tickets to 'Environmental Restoration'
    df_incidents.loc[df_incidents['close_code'].isin(
        ['Capacity Adjustment','Hardware Correction','Redundancy Activation']),'close_code']="Environmental Restoration"

    # plan assignment_group_company secondary analysis
    df_factors.loc[(df_factors['factor']=="company"),"variable_type"] = "analyse2"

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

def create_Xy (df_incidents, df_factors):
    """ subset the columns of df_incidents to those identified as 'analyse' in factors
    Input: dataframe with the incidents, dataframe with the factors, maximum number of factors to consider
    Returns: modified df_incident and df_factors dataframes
    """
    X_columns = df_factors[df_factors['variable_type']=='analyse'].factor
    X = np.array(df_incidents[X_columns])

    y_column = df_factors[df_factors['variable_type']=='response'].factor
    y = np.array (df_incidents[y_column]).squeeze()

    return X, y, X_columns