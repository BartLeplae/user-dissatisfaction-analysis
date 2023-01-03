""" 
Read incident tickets containing survey resonses for the last 365 days from Enterprise Data Lake
Write these tickets to an Excel file
Return the tickets as a dataframe
"""
# Load data with Pyodbc
import pandas as pd
import numpy as np
import pyodbc

# Return cursor result as a dataframe
def as_pandas_DataFrame(cursor):
    "function to return cursor data to a dataframe"
    names = [metadata[0] for metadata in cursor.description]
    return pd.DataFrame([dict(zip(names, row)) for row in cursor], columns=names)


def get_incidents_from_db(
    incident_file: str,
) -> pd.DataFrame:

    # Retrieve the Incidents that contain a customer survey resonse
    # Create connection to EDL through ODBC
    conn = pyodbc.connect(f'DSN=ODBC Impala', autocommit=True)

    # Get cursor to interact with the SQL engine
    cursor = conn.cursor()

    # Select incident tickets for the last year where the user provided a survey response
    # retrieve fields that may be correlated with the survey response
    Query = """
    select close_code, 
    breached_reason_code,
    contact_type, self_service, incident_reopened_flag, 
    sla_result, sla_priority, 
    am_ttr,
    incident_has_ka_related_flag,
    reassignment_count,
    appl_tier, 
    caller_vip, caller_employee_type, 
    survey_response_value,
    ci_name, assignment_group_company, assignment_group_name, kcs_solution
    from datamart_core.dm_incidentcube
    where survey_response_value > 0
    and am_ttr > 0
    and assignment_group_parent in ('PARENT APP MAINTENANCE', 'PARENT APP SERVICES SUPPORT')
    and resolved_date_utc > date_sub(now(),365)"""

    cursor.execute(Query)
    df = as_pandas_DataFrame(cursor) # Convert result set into pandas DataFrame

    # Determine the number of times the knowledge article has been applied
    Query = """
    select kcs_solution, count(*) as ka_count
    from datamart_core.dm_incidentcube where kcs_solution in (
    select distinct kcs_solution
    from datamart_core.dm_incidentcube where assignment_group_parent in ('PARENT APP MAINTENANCE', 'PARENT APP SERVICES SUPPORT')
    and resolved_date_utc > date_sub(now(),365))
    group by kcs_solution
    """

    cursor.execute(Query)
    df_ka_count = as_pandas_DataFrame(cursor) # Convert result set into pandas DataFrame

    conn.close() # Close connection

    # Add the number of times a knowledge article has been used and transform to a log2
    df = pd.merge(df,df_ka_count, on = 'kcs_solution', how='left' )
    df['ka_count'].fillna(0,inplace=True)
    df['ka_count_log'] = np.ceil(np.log2(df['ka_count']+1)/2)
    df.drop(columns='ka_count', inplace=True)

    # Transform time to resolve to logaritmic scale based on number of days
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

    # Write result set to Excel
    df.to_excel(incident_file, index=False)

    return df
