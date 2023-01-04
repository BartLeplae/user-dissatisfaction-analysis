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
from transform_attributes import transform_df_upon_chi2, transform_df_upon_review_values, df_create_dummies

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
    factors_data_file = output_dir / f"factors_1.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    # Transform the data based on review of "factors1.xlsx" and generate "factors2.xlsx"
    df_factors = transform_df_upon_chi2 (df_factors)
    factors_data_file = output_dir / f"factors_2.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    # Evaluate the individual values for each factor and write to "factor_values1.xlsx"
    df_satisfaction = binom_stats(df_incidents, df_factors)
    factor_values_data_file = output_dir / f"factor_values_1.xlsx"
    df_satisfaction.to_excel(factor_values_data_file, index=False)

    # Transform the data upon review of factor_values1.xlsx:
    df_incidents, df_factors = transform_df_upon_review_values(df_incidents, df_factors)

    # For every value, determine the correlation with customer dissatisfaction, write ordered df to factor_values_2.xlsx
    df_satisfaction = binom_stats(df_incidents, df_factors)
    factor_values_data_file = output_dir / f"factor_values_2.xlsx"    
    df_satisfaction.to_excel(factor_values_data_file, index=False)   

    # Create dummies for the fields containing multiple categorical values
    df_incidents, df_factors = df_create_dummies(df_incidents, df_factors)
    factors_data_file = output_dir / f"factors_3.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    # Determine the variables to be used for modelling
    df_modelling_vars = df_factors[df_factors['variable_type'].isin(["analyse"])].factor.values
    print(df_modelling_vars)
    
    for index, row in df_factors[df_factors['variable_type'].isin(["one_hot_encoded"])].iterrows():
        df_one_hot = df_satisfaction[df_satisfaction['factor']==row["factor"]]
        for index2, row2 in df_one_hot.iterrows():
            print(row2["factor"]+"_"+row2["value"])

    
