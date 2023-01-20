""" main:

    - Reads incidents from data lake or from excel file:
    - Statisical correlation between user dissatisfaction and incident attributes
    - Builds regression model: determines expected dissatisfaction ratio against the combination of incident attributes
    - Applies the model for groups of incident tickets: support group, application, ...
    
    to run: python main.py

Attributes:
    - -d to retrieve tickets from the data lake (and create a new excel file for subsequent use)
    - filename of the excel file

Input:
    - Datalake : incidents (when -d attribute is provide)
    - Default Input File: incident_tickets.xlsx (alternative data source when -d is not provided )
    
Output:
    - factors.xlsx: with a list of the factors and their correlation with user dissatisfaction
    - factor_values.xlsx: list of factor - value combinations and their correlation with user dissatisfaction
    - Dissatisfaction Ratio.png: factor - value combinations and their correlation with user dissatisfaction
    - Predicted dissatisfaction_delta.png: factor - value combinations and their correlation with user dissatisfaction
    - Company Dissatisfaction Ratio.png: comparison of user dissatisfaction ratio's and causes per company
    - Group Dissatisfaction Ratio.png: comparison of user dissatisfaction ratio's and causes per support group
    - application Dissatisfaction Ratio.png: comparison of user dissatisfaction ratio's and causes per application
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import argparse
import time
from incidents_from_odbc import get_incidents_from_db
from stats import chi2_stats, binom_stats
from transform_attributes import transform_df_upon_chi2, transform_df_upon_review_values, df_create_dummies, create_Xy
from model import DecisionTree
import matplotlib.pyplot as plt
import seaborn as sns

def get_project_root() -> Path:
    """Get the root of the current project."""
    return Path(__file__).parent.parent

sys.path.append(Path(__file__).parent.parent.parent.__str__())   # Fix for 'no module named src' error

if __name__ == "__main__":

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
        
    # Perfrom chi2 test to identify the relevant factors (columns) and write the factors to an excel file for further manua analysis
    df_factors = chi2_stats(df_incidents)
    factors_data_file = output_dir / f"factors.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    # Transform the data based on a manual review of "factors1.xlsx"
    df_factors = transform_df_upon_chi2 (df_factors)
    factors_data_file = output_dir / f"factors.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    # List the individual values for each factor along with their correlation with user dissatisfaction and write to "factor_values1.xlsx"
    df_factor_values = binom_stats(df_incidents, df_factors)
    df_factor_values = pd.merge(df_factors, df_factor_values, on = 'factor', how='right')
    df_factor_values.sort_values(by=['chi', 'factor','dissatisfied_ratio'],ascending=[False,True,False],inplace=True)
    factor_values_data_file = output_dir / f"factor_values.xlsx"
    df_factor_values.to_excel(factor_values_data_file, index=False)

    # Transform the incident data upon review of factor_values1.xlsx:
    df_incidents, df_factors = transform_df_upon_review_values(df_incidents, df_factors)

    # For every value, determine the correlation with customer dissatisfaction after transformation, write ordered df to factor_values_2.xlsx
    df_factor_values = binom_stats(df_incidents, df_factors)
    df_factor_values = pd.merge(df_factors, df_factor_values, on = 'factor', how='right')
    df_factor_values.sort_values(by=['chi', 'factor','dissatisfied_ratio'],ascending=[False,True,False],inplace=True)
    factor_values_data_file = output_dir / f"factor_values.xlsx"    
    df_factor_values.to_excel(factor_values_data_file, index=False)   

    # Create dummies for the fields containing multiple categorical values, write ordered df to factor_3.xlsx
    df_incidents, df_factors = df_create_dummies(df_incidents, df_factors)
    factors_data_file = output_dir / f"factors.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    # Create X and y with maximum of Z factors and apply to DecisionTree (used as regression model)
    X, y, X_columns = create_Xy(df_incidents, df_factors)
    df_factors_num = pd.DataFrame({'factor': X_columns, 'colnum': range(len(X_columns))})
    df_factors = pd.merge(df_factors_num, df_factors, on = 'factor', how='right') #add column number as attribute

    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

    # Create DecisionTree model based on X and y
    model = DecisionTree(X,y, X_columns.values)
    print(model)

    # add the model feature importances to df_factors and write to factors_4.xlsx
    df_model_features = pd.DataFrame(data={'factor': X_columns, 'feature_importance': model.feature_importances_})
    df_factors = pd.merge(df_factors, df_model_features, on = 'factor', how='left')
    df_factors.sort_values(by=['feature_importance','chi','p'],ascending=[False,False,True],inplace=True)
    factors_data_file = output_dir / f"factors.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    # For every value, determine the correlation with customer dissatisfaction after transformation, write ordered df to factor_values_3.xlsx
    df_factor_values = binom_stats(df_incidents, df_factors)
    df_factor_values = pd.merge(df_factors, df_factor_values, on = 'factor', how='right')

    # Compute the average predicted satisfaction rating
    df_incidents['dissatisfaction_proba'] = model.predict_proba(X)[:,1]
    avg_dissatisfaction = df_incidents['dissatisfaction_proba'].mean()
    print(avg_dissatisfaction)

    # For every factor - value combination: compute the predicted satisfaction rating if this value would be enforced
    df_factor_values['predicted_dissatisfaction'] = avg_dissatisfaction
    df_factor_values['predicted_dissatisfaction_delta'] = 0

    for index, row in df_factor_values.iterrows():
        Z = np.copy(X)
        Z[:,int(row['colnum'])] = row['value']
        new_col = 'pred_'+ row['factor'] + '_' + str(row['value'])
        df_incidents[new_col] = model.predict_proba(Z)[:,1]
        # df_factor_values.loc[index,'predicted_dissatisfaction']=df_incidents[new_col].mean()
        df_factor_values.loc[index,'predicted_dissatisfaction_delta']=df_incidents[new_col].mean() -avg_dissatisfaction
        df_incidents[new_col] = df_incidents[new_col] - df_incidents['dissatisfaction_proba']

    df_factor_values["factor_value"] =  df_factor_values["factor"] + ": " + df_factor_values["value"].astype(str)
    df_factor_values.sort_values(by=['feature_importance','chi', 'factor','value'],ascending=[False,False,True,True],inplace=True)
    factor_values_data_file = output_dir / f"factor_values.xlsx"    
    df_factor_values.to_excel(factor_values_data_file, index=False)   

    #create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(x=df_factor_values.dissatisfied_ratio, y=df_factor_values.factor_value, orient='h')
    plt.axvline(avg_dissatisfaction)
    plt.title('Dissatisfaction Ratio')
    plt.xlabel('Dissatisfaction %')
    plt.ylabel('Factor + Value')
    plt.tight_layout()
    dissatisfaction_ratio_file = output_dir / f"Dissatisfaction Ratio.png"    
    plt.savefig(dissatisfaction_ratio_file, dpi=300)
    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(x=df_factor_values.predicted_dissatisfaction_delta, y=df_factor_values.factor_value, orient='h')
    plt.axvline(0)
    plt.title('Dissatisfaction Delta')
    plt.xlabel('Dissatisfaction Delta %')
    plt.ylabel('Factor + Value')
    plt.tight_layout()
    dissatisfaction_dissatisfaction_delta_file = output_dir / f"Predicted dissatisfaction_delta.png"    
    plt.savefig(dissatisfaction_dissatisfaction_delta_file, dpi=300)

    # Write an ordered list of the most impactful factors
    # "predicted_dissatisfaction_delta" is the predicted reduction in disatisfaction if the factor is eliminated (value associated with the factor = 0)
    # we are interested in factors that increase dissatisfaction: so look for negative values and eliminate the - sign these for reporting
    df_most_impactful_factors = df_factor_values[(df_factor_values["predicted_dissatisfaction_delta"]<0) & (df_factor_values["value"]==0)].copy()
    df_most_impactful_factors["predicted_dissatisfaction_delta"] = df_most_impactful_factors["predicted_dissatisfaction_delta"].apply(lambda x: x*-1)
    df_most_impactful_factors.sort_values(by="predicted_dissatisfaction_delta", ascending=False, inplace=True)

    # Merge "predicted_dissatisfaction_delta" with the factors
    df_factors = pd.merge(df_factors, df_most_impactful_factors[["factor","predicted_dissatisfaction_delta"]], on = 'factor', how='left')
    df_factors.sort_values(by=['predicted_dissatisfaction_delta','feature_importance','chi','p'],ascending=[False,False,False,True],inplace=True)
    factors_data_file = output_dir / f"factors.xlsx"
    df_factors.to_excel(factors_data_file,index=False)

    # Write the predicted effects for all incident tickets to "incidents_dissatisfaction_pred.xlsx"    
    # incident_result_data_file = output_dir / f"incidents_dissatisfaction_pred.xlsx"    
    # df_incidents.to_excel(incident_result_data_file, index=False)

    # Create graph with a comparison of user dissatisfaction per supporting company and corresponding causal factors
    # Limit to support companies with more than 1000 survey responses
    company_analysis_avg = pd.pivot_table(data=df_incidents, index="company", 
                        values=["user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"],
                        aggfunc='mean'
                        )
    company_analysis_avg.reset_index(inplace=True)

    company_analysis_count = pd.pivot_table(data=df_incidents, index="company", 
                        values=["application"],
                        aggfunc='count'
                        )
    company_analysis_count.reset_index(inplace=True)

    company_analysis = pd.merge(company_analysis_count, company_analysis_avg)
    company_analysis = company_analysis[company_analysis["application"]>1000]
    company_analysis = company_analysis[["company","user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"]].set_index('company')
    company_analysis.sort_values(by="user_dissatisfied", inplace=True, ascending=False)
    company_analysis.columns=["dissatisfaction%","reopened","resolution_time","no_resolution"]
    company_analysis.plot(kind='barh', stacked=True )
    
    plt.axvline(avg_dissatisfaction, color='r')
    company_dissatisfaction_ratio_file = output_dir / f"Company Dissatisfaction Ratio.png"  
    plt.savefig(company_dissatisfaction_ratio_file, dpi=300) 

    # Create graph with a comparison of user dissatisfaction per support group and corresponding causal factors
    # Limit to support groups with more than 200 survey responses
    group_analysis_avg = pd.pivot_table(data=df_incidents, index="group", 
                        values=["user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"],
                        aggfunc='mean'
                        )
    group_analysis_avg.reset_index(inplace=True)

    group_analysis_count = pd.pivot_table(data=df_incidents, index="group", 
                        values=["application"],
                        aggfunc='count'
                        )
    group_analysis_count.reset_index(inplace=True)
    group_analysis = pd.merge(group_analysis_count, group_analysis_avg)
    group_analysis = group_analysis[group_analysis["application"]>200]
    group_analysis = group_analysis[["group","user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"]].set_index('group')
    group_analysis.sort_values(by="user_dissatisfied", inplace=True, ascending=False)
    group_analysis.columns=["dissatisfaction%","reopened","resolution_time","no_resolution"]
    group_analysis.plot(kind='barh', stacked=True )
    plt.axvline(avg_dissatisfaction, color='r')
    group_dissatisfaction_ratio_file = output_dir / f"Group Dissatisfaction Ratio.png"  
    plt.savefig(group_dissatisfaction_ratio_file, dpi=300) 
    
    # Create graph with a comparison of user dissatisfaction per application and corresponding causal factors
    # Limit to applications with more than 150 survey responses
    application_analysis_avg = pd.pivot_table(data=df_incidents, index="application", 
                        values=["user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"],
                        aggfunc='mean'
                        )
    application_analysis_avg.reset_index(inplace=True)

    application_analysis_count = pd.pivot_table(data=df_incidents, index="application", 
                        values=["group"],
                        aggfunc='count'
                        )
    application_analysis_count.reset_index(inplace=True)
    application_analysis = pd.merge(application_analysis_count, application_analysis_avg)
    application_analysis = application_analysis[application_analysis["group"]>150]
    application_analysis = application_analysis[["application","user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"]].set_index('application')
    application_analysis.sort_values(by="user_dissatisfied", inplace=True, ascending=False)
    application_analysis.columns=["dissatisfaction%","reopened","resolution_time","no_resolution"]
    application_analysis.plot(kind='barh', stacked=True )
    plt.axvline(avg_dissatisfaction, color='r')
    application_dissatisfaction_ratio_file = output_dir / f"application Dissatisfaction Ratio.png"  
    plt.savefig(application_dissatisfaction_ratio_file, dpi=300) 



    
