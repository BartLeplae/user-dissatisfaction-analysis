import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stats import chi2_stats, ratio_stats, binom_stats

def plot_factor_values(df_factor_values, avg_dissatisfaction, output_dir):
    #create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(x=df_factor_values.dissatisfied_ratio, y=df_factor_values.factor_value, orient='h')
    plt.axvline(avg_dissatisfaction)
    plt.title('Dissatisfaction Ratio')
    plt.xlabel('Dissatisfaction %')
    plt.ylabel('Factor + Value')
    plt.tight_layout()
    dissatisfaction_ratio_file = output_dir / f"05 Dissatisfaction Ratio.png"    
    plt.savefig(dissatisfaction_ratio_file, dpi=300)
    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(x=df_factor_values.predicted_dissatisfaction_delta, y=df_factor_values.factor_value, orient='h')
    plt.axvline(0)
    plt.title('Dissatisfaction Delta')
    plt.xlabel('Dissatisfaction Delta %')
    plt.ylabel('Factor + Value')
    plt.tight_layout()
    dissatisfaction_dissatisfaction_delta_file = output_dir / f"06 Predicted dissatisfaction_delta.png"    
    plt.savefig(dissatisfaction_dissatisfaction_delta_file, dpi=300)


def create_ordered_excel(df_incidents, index_group, avg_dissatisfaction, output_file):
    # Create Excel with a comparison of user dissatisfaction per application and corresponding causal factors
    # Sort by statical relevance and flag the most relevant ones
    org_names = ["contact_type","dissatisfied count","user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"]
    org_names = index_group + org_names
    new_names = ["total count","dissatisfied count","dissatisfaction%","reopened","resolution_time","no_resolution"]
    new_names = index_group + new_names

    application_analysis_avg = pd.pivot_table(
                        data=df_incidents, 
                        index=index_group,
                        values=["user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"],
                        aggfunc='mean'
                        )
    application_analysis_avg.reset_index(inplace=True)

    application_analysis_count = pd.pivot_table(
                        data=df_incidents, 
                        index=index_group, 
                        values=["contact_type"],
                        aggfunc='count'
                        )
    application_analysis_count.reset_index(inplace=True)
    application_analysis = pd.merge(application_analysis_count, application_analysis_avg)
    application_analysis["dissatisfied count"] = application_analysis["contact_type"]*application_analysis["user_dissatisfied"]
    application_analysis = application_analysis[org_names]
    application_analysis.columns=new_names

    #identify pvalue for satisfaction rating = 1/2 of overall average, relevance level = 5%, clip to min 5 dissatisfied
    application_analysis = binom_stats(application_analysis, avg_dissatisfaction/2,0.05,5) 

    #write sorted file
    application_analysis.sort_values(by=["relevant","pvalue","dissatisfied count","total count"], ascending=[False,True, False, True], inplace=True)
    application_analysis.to_excel(output_file)

def write_ordered_plot(df_incidents, index_group, avg_dissatisfaction, output_file, title, limit):
        # Create graph with a comparison of user dissatisfaction per supporting company and corresponding causal factors
    # Limit to support companies with more than 1000 survey responses
    org_names = ["user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"]
    org_names = index_group + org_names
    new_names = ["dissatisfaction%","reopened","resolution_time","no_resolution"]
    new_names = index_group + new_names
    company_analysis_avg = pd.pivot_table(data=df_incidents, 
                        index=index_group, 
                        values=["user_dissatisfied","pred_reopened_0.0","pred_days_to_resolve_0.0","pred_close_code_No Resolution Action_0.0"],
                        aggfunc='mean'
                        )
    company_analysis_avg.reset_index(inplace=True)

    company_analysis_count = pd.pivot_table(data=df_incidents, 
                        index=index_group, 
                        values=["contact_type"],
                        aggfunc='count'
                        )
    company_analysis_count.reset_index(inplace=True)

    company_analysis = pd.merge(company_analysis_count, company_analysis_avg)
    company_analysis = company_analysis[company_analysis["contact_type"]>limit]
    company_analysis = company_analysis[org_names]
    company_analysis.columns=new_names
    company_analysis = company_analysis.set_index(index_group)
    company_analysis.sort_values(by="dissatisfaction%", inplace=True, ascending=False)

    company_analysis.plot(kind='barh', stacked=True ) 

    plt.axvline(avg_dissatisfaction, color='r')
    plt.axvline(0, color='grey')
    plt.title(title)
    plt.savefig(output_file, dpi=300) 
    