""" main:
    - Reads incidents from data lake or from a csv file:
    - Determines statisical correlation between user dissatisfaction and incident attributes
    - Builds regression model: determines expected dissatisfaction ratio against the combination of incident attributes
    - Applies the model to the tickets with and without survey responses
    
    to run: python main.py

Attributes:
    - -d to retrieve tickets from the data lake (and create a new csv file for subsequent use)
    - filename of the excel file

Input:
    - Datalake : incidents (when -d attribute is provided)
    - Default Input File: incident_tickets.xlsx (alternative data source when -d is not provided )
    
Output:
    - Several Excel files in the 'out' folder: factors, factor_values, support company, support group, application
    - Several png files (graphs) in the 'out' folder
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import argparse

from incidents_from_odbc import get_incidents_from_db, get_all_incidents_from_db
from stats import chi2_stats, ratio_stats, binom_stats
from output import plot_factor_values, create_ordered_excel, write_ordered_plot, write_response_ratio_plot
from transform_attributes import transform_df_upon_chi2, transform_df_upon_review_values, df_create_dummies, create_Xy
from model import DecisionTree


def get_project_root() -> Path:
    """Get the root of the current project."""
    return Path(__file__).parent.parent

sys.path.append(Path(__file__).parent.parent.parent.__str__())   # Fix for 'no module named src' error

if __name__ == "__main__":

    # Define a parser for comand line operation
    parser = argparse.ArgumentParser(description="User Dissatisfaction Analysis",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-incidents_fname', default="incident_tickets", help="CSV file with Incident data", )
    parser.add_argument('-d', '--db', help="read incident data from database", action='store_true')
    args = parser.parse_args()

    # Define the paths to be used
    project_path = get_project_root()
    data_dir = project_path / "data"
    output_dir = project_path / "out"

    factors_data_file = output_dir / f"00 factors.xlsx"
    factor_values_data_file = output_dir / f"01 factor_values.xlsx"
    factor_values_data_file_initial = output_dir / f"01 initial_factor_values.xlsx"

    # Create dataframe with the incidents either from the database or Excel file
    incident_data_file = data_dir / f"{args.incidents_fname}.csv"
    all_incidents_data_file = data_dir / f"all_incidents.csv"
    
    if (args.db): 
        print("Read incidents from database and store in", incident_data_file)
        df_incidents = get_incidents_from_db(incident_data_file)
        df_all_incidents = get_all_incidents_from_db(all_incidents_data_file)
    else:
        print("Read incidents from ", incident_data_file)
        df_incidents = pd.read_csv(incident_data_file)
        df_all_incidents = pd.read_csv(all_incidents_data_file)
        
    # Perfrom chi2 test to identify the relevant factors (columns) and write the factors to an excel file for further manual analysis
    df_factors = chi2_stats(df_incidents)
    df_factors.to_excel(factors_data_file,index=False)

    # Transform the data based on a manual review of the factors file
    df_factors = transform_df_upon_chi2 (df_factors)
    df_factors.to_excel(factors_data_file,index=False)

    # List the individual values for each factor along with their correlation with user dissatisfaction and write to "01 initial_factor_values.xlsx"
    df_factor_values = ratio_stats(df_incidents, df_factors)
    df_factor_values = pd.merge(df_factors, df_factor_values, on = 'factor', how='right')
    df_factor_values.sort_values(by=['chi', 'factor','value'],ascending=[False,True,True],inplace=True)
    df_factor_values.to_excel(factor_values_data_file_initial, index=False)

    # Transform the incident data upon review of "01 factor_values.xlsx":
    df_incidents, df_factors = transform_df_upon_review_values(df_incidents, df_factors)

    # For every value, determine the correlation with customer dissatisfaction after transformation, write ordered df to "01 initial_factor_values.xlsx"
    df_factor_values = ratio_stats(df_incidents, df_factors)
    df_factor_values = pd.merge(df_factors, df_factor_values, on = 'factor', how='right')
    df_factor_values.sort_values(by=['chi', 'factor','value'],ascending=[False,True,True],inplace=True)
    df_factor_values.to_excel(factor_values_data_file, index=False)   

    # Create dummies for the fields containing multiple categorical values, write ordered df to factor.xlsx
    df_incidents, df_factors = df_create_dummies(df_incidents, df_factors)
    df_factors.to_excel(factors_data_file,index=False)

    # Create X and y with maximum of Z factors and apply to DecisionTree (used as regression model)
    X, y, X_columns = create_Xy(df_incidents, df_factors)
    df_factors_num = pd.DataFrame({'factor': X_columns, 'colnum': range(len(X_columns))})
    df_factors = pd.merge(df_factors_num, df_factors, on = 'factor', how='right') #add column number as attribute

    # Create DecisionTree model based on X and y
    model = DecisionTree(X,y)
    print(model)

    # add the model feature importances to df_factors and write to factors.xlsx
    df_model_features = pd.DataFrame(data={'factor': X_columns, 'feature_importance': model.feature_importances_})
    df_factors = pd.merge(df_factors, df_model_features, on = 'factor', how='left')
    df_factors.sort_values(by=['feature_importance','chi','p'],ascending=[False,False,True],inplace=True)

    df_factors.to_excel(factors_data_file,index=False)

    # For every value, determine the correlation with customer dissatisfaction after transformation, write ordered df to "01 factor_values.xlsx"
    df_factor_values = ratio_stats(df_incidents, df_factors)
    df_factor_values = pd.merge(df_factors, df_factor_values, on = 'factor', how='right')

    # Compute the predicted satisfaction rating across all incident records
    df_incidents['dissatisfaction_proba'] = model.predict_proba(X)[:,1]
    avg_dissatisfaction = df_incidents['dissatisfaction_proba'].mean()
    print(avg_dissatisfaction)

    # For every factor - value combination: compute the predicted satisfaction rating if this value would have been enforced
    # e.g predict how much customer satisfaction would change if all incidents would be resolved the same day, in 1 day, in 2 days, ...

    df_factor_values['predicted_dissatisfaction'] = avg_dissatisfaction
    df_factor_values['predicted_dissatisfaction_delta'] = 0

    for index, row in df_factor_values.iterrows():
        Z = np.copy(X)
        Z[:,int(row['colnum'])] = row['value'] # replace the actual value with the value for which we want to predict the effect
        new_col = 'pred_'+ row['factor'] + '_' + str(row['value']) 
        df_incidents[new_col] = model.predict_proba(Z)[:,1] # predict
        df_factor_values.loc[index,'predicted_dissatisfaction_delta']=df_incidents[new_col].mean() - avg_dissatisfaction # calculate the difference in satisfaction rating
        df_incidents[new_col] = df_incidents[new_col] - df_incidents['dissatisfaction_proba'] # replace with the predicted satisfaction with the difference in satisfaction

    df_factor_values["factor_value"] =  df_factor_values["factor"] + ": " + df_factor_values["value"].astype(str) # factor value: combination for reporting purposes
    df_factor_values.sort_values(by=['feature_importance','chi', 'factor','value'],ascending=[False,False,True,True],inplace=True)

    
    # Write the predicted values to Excel and plot for analysis purposes
    df_factor_values.to_excel(factor_values_data_file, index=False)   
    plot_factor_values(df_factor_values, avg_dissatisfaction, output_dir)

    # Create an ordered list of the most impactful factors
    # "predicted_dissatisfaction_delta" is the predicted reduction in disatisfaction if the factor is eliminated (value associated with the factor = 0)
    # we are interested in factors that increase dissatisfaction: so look for negative values and eliminate the - sign these for reporting
    df_most_impactful_factors = df_factor_values[(df_factor_values["predicted_dissatisfaction_delta"]<0) & (df_factor_values["value"]==0)].copy()
    df_most_impactful_factors["predicted_dissatisfaction_delta"] = df_most_impactful_factors["predicted_dissatisfaction_delta"].apply(lambda x: x*-1)
    df_most_impactful_factors.sort_values(by="predicted_dissatisfaction_delta", ascending=False, inplace=True)

    # Merge "predicted_dissatisfaction_delta" with the factors
    df_factors = pd.merge(df_factors, df_most_impactful_factors[["factor","predicted_dissatisfaction_delta"]], on = 'factor', how='left')
    df_factors.sort_values(by=['predicted_dissatisfaction_delta','feature_importance','chi','p'],ascending=[False,False,False,True],inplace=True)
    df_factors.to_excel(factors_data_file,index=False)

    # Write Excel files for Company, Company+Group, Company+Group+Application ordered by statistical relevance 
    create_ordered_excel(df_incidents, ["company"], avg_dissatisfaction, output_dir / f"10 Support Company Dissatisfaction.xlsx")
    create_ordered_excel(df_incidents, ["company","group"], avg_dissatisfaction, output_dir / f"11 Support Group Dissatisfaction.xlsx")
    create_ordered_excel(df_incidents, ["company","group","application"], avg_dissatisfaction, output_dir / f"12 Application Dissatisfaction.xlsx")

    # Write barcharts for company, group and application
    write_ordered_plot(df_incidents, ["company"], avg_dissatisfaction, output_dir / f"51 Support Company Dissatisfaction.png","Companies",1000)
    write_ordered_plot(df_incidents, ["group"], avg_dissatisfaction, output_dir / f"52 Support Group Dissatisfaction.png","Groups",200)
    write_ordered_plot(df_incidents, ["application"], avg_dissatisfaction, output_dir / f"53 Support App Dissatisfaction.png","Applications",150)    

    # Plot barcharts for each of the differentiating attributes 
    write_ordered_plot(df_incidents, ["close_code_Information Provided / Training"], avg_dissatisfaction, output_dir / f"20 Information Provided Dissatisfaction.png","Close Code: Information Provided?",150)    
    write_ordered_plot(df_incidents, ["reassignment_count"], avg_dissatisfaction, output_dir / f"21 Reassignment Dissatisfaction.png","Ticket Reassignment Count",150)    
    write_ordered_plot(df_incidents, ["caller_is_employee"], avg_dissatisfaction, output_dir / f"22 Employee Dissatisfaction.png","Reported by Employee? (vs. External)",150)
    write_ordered_plot(df_incidents, ["has_knowledge_article"], avg_dissatisfaction, output_dir / f"23 Knowledge Article Dissatisfaction.png","Ticket has knowledge article?",150)
    write_ordered_plot(df_incidents, ["close_code_Data Correction"], avg_dissatisfaction, output_dir / f"24 Data Correction Dissatisfaction.png","Close Code: Data Correction?",150)
    write_ordered_plot(df_incidents, ["sla_breached"], avg_dissatisfaction, output_dir / f"25 SLA Breached Dissatisfaction.png","SLA Breached?",150)
    write_ordered_plot(df_incidents, ["self_service"], avg_dissatisfaction, output_dir / f"26 Self Service Dissatisfaction.png","Self Service?",150)
    write_ordered_plot(df_incidents, ["priority_is_4"], avg_dissatisfaction, output_dir / f"27 Priority 4 Dissatisfaction.png","Priority 4 (versus 1, 2 or 3)",150)
    write_ordered_plot(df_incidents, ["close_code_Reboot / Restart"], avg_dissatisfaction, output_dir / f"28 Reboot Dissatisfaction.png","Close Code: Reboot, Restart",150)
    write_ordered_plot(df_incidents, ["close_code_Security Modification"], avg_dissatisfaction, output_dir / f"29 Security Modification Dissatisfaction.png","Close Code: Security Modification",150)
    write_ordered_plot(df_incidents, ["close_code_Software Correction"], avg_dissatisfaction, output_dir / f"30 Software Correction Dissatisfaction.png","Close Code: Software Correction",150)
    write_ordered_plot(df_incidents, ["close_code_Environmental Restoration"], avg_dissatisfaction, output_dir / f"31 Environmental Restoration Dissatisfaction.png","Close Code: Environmental Restoration",150)    

    # Read all of the incidents (those with and those without survey responses)
    # Create a new simplified DecisionTree model (only based on the 3 most determining factors)
    df_all_incidents_responded = df_all_incidents[df_all_incidents["user_responded"]==1]
    X = np.array(df_all_incidents_responded[["reopened","days_to_resolve","no resolution"]])
    y = np.array (df_all_incidents_responded["user_dissatisfied"]).squeeze()
    model_all_incidents = DecisionTree(X,y)
    print(model_all_incidents)

    # Apply the simplified model on all incident tickets
    X = np.array(df_all_incidents[["reopened","days_to_resolve","no resolution"]])
    df_all_incidents["dissatisfied_proba"] = model_all_incidents.predict_proba(X)[:,1]

    # Average predicted dissatisfaction
    avg_pred_dissatisfaction_all = df_all_incidents['dissatisfied_proba'].mean()
    print(avg_pred_dissatisfaction_all)

    Z = np.copy(X)
    Z[:,int(0)] = 0 # No tickets repened
    df_all_incidents["pred_reopened_0.0"] = model_all_incidents.predict_proba(Z)[:,1] -  df_all_incidents['dissatisfied_proba']

    Z = np.copy(X)
    Z[:,int(1)] = 0 # Resolved on day 0
    df_all_incidents["pred_days_to_resolve_0.0"] = model_all_incidents.predict_proba(Z)[:,1] - df_all_incidents['dissatisfied_proba']

    Z = np.copy(X)
    Z[:,int(2)] = 0 # no tickets without resolution
    df_all_incidents["pred_close_code_No Resolution Action_0.0"] = model_all_incidents.predict_proba(Z)[:,1] - df_all_incidents['dissatisfied_proba']
    
    df_all_incidents["user_dissatisfied"] = df_all_incidents["dissatisfied_proba"] # We don't have actual dissatisfaction information - use predicted values
    df_all_incidents["contact_type"] = 1 # Contact type is used to count the records in write_ordered_plot

    # Plot the result, differentiated by user_reponse
    write_ordered_plot(df_all_incidents, ["user_responded"], avg_pred_dissatisfaction_all, output_dir / f"08 User Responded Dissatisfaction.png","Dissatisfaction% - User entered survey?",0) 

    # Plot the survey response ratios
    write_response_ratio_plot(df_all_incidents, output_dir / f"07 Survey Response Ratio.png")

