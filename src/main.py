""" main:

    - Reads incidents from data lake or from excel file:
    - Statisical correlation between user dissatisfaction and incident attributes
    - Builds regression model: determines expected dissatisfaction ration against the combination of incident attributes
    - Applies the model for groups of incident tickets: support group, application, ...
    
    to run: python main.py

Attributes:
    - -d to retrieve tickets from the data lake (and create a new excel file for subsequent use)
    - filename of the excel file

Input:
    - Datalake : incidents
    - Regression Model
    
Output:
    - Excel file with ...
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
import argparse
import time
from incidents_from_odbc import get_incidents_from_db
from stats import chi2_stats, binom_stats

def get_project_root() -> Path:
    """Get the root of the current project."""
    return Path(__file__).parent.parent

sys.path.append(Path(__file__).parent.parent.parent.__str__())   # Fix for 'no module named src' error

if __name__ == "__main__":
    # Ignore annoying warning
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    # warnings.simplefilter(action='ignore', category=ConvergenceWarning)

    # Make timestamp for timing
    global_start = time.time()
    start = time.time()  # Before timestamp

    # Define a parser for comand line operation
    parser = argparse.ArgumentParser(description="User Dissatisfaction Analysis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-incidents_fname', default="incident_tickets", help="Excel file with Incident data", )
    parser.add_argument('-d', '--db', help="read incident data from database", action='store_true')
    args = parser.parse_args()

    # Define the paths to be used
    project_path = get_project_root()
    data_dir = project_path / "data"
    output_dir = project_path / "out"

    # Create dataframe with the incidents either from the database or Excel file
    incident_data_file = data_dir / f"{args.incidents_fname}.xlsx"
    
    if (args.db):
        print("Read incidents from database and store in", incident_data_file)
        df_incidents = get_incidents_from_db(incident_data_file)
    else:
        print("Read incidents from ", incident_data_file)
        df_incidents = pd.read_excel(incident_data_file)
        
    # Perfrom chi2 test to identify the relevant factors (columns) and write the factors to an excel file for further analysis
    df_factors = chi2_stats(df_incidents)
    
    # Ignore the factors for which the p value is greater than 5%
    df_factors.loc[(df_factors['variable_type']=="analyse")&(df_factors['p']>0.05),"variable_type"] = "ignore"

    # Perform secondary analysis when number of values > 20 (application, resolving group, ...)
    df_factors.loc[(df_factors['variable_type']=="analyse")&(df_factors['unique_values']>20),"variable_type"] = "analyse2"  

    # Write factors to Excel
    factors_data_file = output_dir / f"factors.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    # Write factor values to Excel
    df_satisfaction = binom_stats(df_incidents, df_factors)
    factor_values_data_file = output_dir / f"factor_values.xlsx"
    df_satisfaction.to_excel(factor_values_data_file, index=False)

    # UPON review of factor_values.xlsx:
    # Considering the similarities in dissatisfaction ratio and limited total, limit ttr_days_log to 5 
    df_incidents.loc[df_incidents['ttr_days_log']>5,'ttr_days_log']=5

    # Ignore the ttr_days for subsequent analysis
    df_factors.loc[(df_factors['factor']=="ttr_days"),"variable_type"] = "ignore"

    # Add sla_result == "Unknown" to "Achieved" given the low number of records
    df_incidents.loc[df_incidents['sla_result']=="Unknown",'sla_result']="Achieved"

    # Limit reassignment count to given the low values for higher reassignment counts
    df_incidents.loc[df_incidents['reassignment_count']>4,'reassignment_count']=4

    # Reclassify close codes with less than 150 tickets to 'Environmental Restoration'
    df_incidents.loc[df_incidents['close_code'].isin(
        ['Capacity Adjustment','Hardware Correction','Redundancy Activation']),'close_code']="Environmental Restoration"

    # plan assignment_group_company secondary analysis
    df_factors.loc[(df_factors['factor']=="assignment_group_company"),"variable_type"] = "analysis2"

    # Reclassify sla_priority 'VIP' to 'Priority 3' given the low number of records
    df_incidents.loc[df_incidents['sla_priority']=="VIP",'sla_priority']="Priority 3"

    # Ignore the ka_count_log for subsequent analysis given the lack of correlation with dissatisfaction
    df_factors.loc[(df_factors['factor']=="ka_count_log"),"variable_type"] = "ignore"

    # Ignore the contact_type for subsequent analysis since 'self_service' is a better differentiator
    df_factors.loc[(df_factors['factor']=="contact_type"),"variable_type"] = "ignore"

    # Ignore the breached_reason_code for subsequent analysis since values are insufficiently differentiated or have low occurences
    df_factors.loc[(df_factors['factor']=="breached_reason_code"),"variable_type"] = "ignore"

    # Ignore the appl_tier for subsequent analysis since values are insufficiently differentiated or have low occurences
    df_factors.loc[(df_factors['factor']=="appl_tier"),"variable_type"] = "ignore"

    # Write adjusted factor values to new Excel file
    df_satisfaction = binom_stats(df_incidents, df_factors)
    factor_values_data_file = output_dir / f"factor_values_2.xlsx"    
    df_satisfaction.to_excel(factor_values_data_file, index=False)   
