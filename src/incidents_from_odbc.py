""" 
Read incident tickets containing survey resonses for the last 365 days from Enterprise Data Lake
Write these tickets to an Excel file
Return the tickets as a dataframe
"""
# Load data with Pyodbc
import pandas as pd
import numpy as np
import pyodbc
from transform_attributes import transform_df_upon_db_retrieval

# Return cursor result as a dataframe
def as_pandas_DataFrame(cursor):
    "function to return cursor data to a dataframe"
    names = [metadata[0] for metadata in cursor.description]
    return pd.DataFrame([dict(zip(names, row)) for row in cursor], columns=names)


def get_incidents_from_db(
    incident_file: str,
) -> pd.DataFrame:
    """ Retrieve the Incidents that contain a customer survey resonse from the data lake
        Create connection to EDL through ODBC
    Input:
        - File to which to store the retrieved data
    Output:
        - Dataframe with the data retrieved from the data lake
    """

    # Connect through ODBC as defined on the machine where this code is run
    conn = pyodbc.connect(f'DSN=ODBC Impala', autocommit=True)

    # Get cursor to interact with the SQL engine
    cursor = conn.cursor()

    # Select incident tickets for the last year where the user provided a survey response
    # retrieve fields that may be correlated with the survey response
    Query = """
    select close_code, 
    breached_reason_code,
    contact_type, self_service, incident_reopened_flag reopened, 
    sla_result, sla_priority, 
    am_ttr,
    incident_has_ka_related_flag has_knowledge_article,
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

    # Transform the dataframe
    df = transform_df_upon_db_retrieval (df)

    # Write result set to Excel
    df.to_excel(incident_file, index=False)

    return df
