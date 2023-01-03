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
from chi2_stats import chi2_stats

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
    factors_data_file = output_dir / f"factors.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    


