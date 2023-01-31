import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stats import chi2_stats, ratio_stats, binom_stats

def plot_factor_values(df_factor_values, avg_dissatisfaction, output_dir):
    #create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(x=100*df_factor_values.dissatisfied_ratio, y=df_factor_values.factor_value, orient='h')
    plt.axvline(avg_dissatisfaction)
    plt.title('Dissatisfaction Ratio')
    plt.xlabel('Dissatisfaction %')
    plt.ylabel('Factor + Value')
    ax.bar_label(ax.containers[0], fmt='%.1f%%', padding=3)
    plt.tight_layout()
    dissatisfaction_ratio_file = output_dir / f"05 Dissatisfaction Ratio.png"    
    plt.savefig(dissatisfaction_ratio_file, dpi=300)
    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(x=100*df_factor_values.predicted_dissatisfaction_delta, y=df_factor_values.factor_value, orient='h')
    plt.axvline(0)
    plt.axvline(-10, color="white")
    plt.axvline(70, color="white")
    plt.title('Difference in satisfaction when this value would be applied')
    plt.xlabel('Dissatisfaction Delta %')
    plt.ylabel('Factor + Value')
    ax.bar_label(ax.containers[0], fmt='%.1f%%', padding=3)
    plt.tight_layout()
    dissatisfaction_dissatisfaction_delta_file = output_dir / f"06 Predicted dissatisfaction_delta.png"    
    plt.savefig(dissatisfaction_dissatisfaction_delta_file, dpi=300)


def create_ordered_excel(df_incidents, index_group, avg_dissatisfaction, output_file):
    """ Create Excel with a comparison of user dissatisfaction per application and corresponding causal factors
    Sort by statical relevance and flag the most relevant ones
    Input:  dataframe with incident tickets
            index_group: variables to be used as index (rows) in the pivot_table
    Returns: None
    """
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
    """ Create horizontal barchart with a comparison of user dissatisfaction per given index_group and corresponding attributes
        Limit to support companies with more than 1000 survey responses
    Input:  dataframe with incident tickets
            index_group: variables to be used as index (rows) in the pivot_table
            avg_dissatisfaction: draw vertical line on horizontal barplot with the average dissatisfaction
            output_file: file to be created
            title: to be displayed on top of the bargraph
            limit: only show those factor-value combination that have more than the 'limit' number of tickets
    Returns: None
    """
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
    company_analysis["dissatisfaction%"] *= 100
    company_analysis["reopened"] *= 100
    company_analysis["resolution_time"] *= 100
    company_analysis["no_resolution"] *= 100
    company_analysis.sort_values(by="dissatisfaction%", inplace=True, ascending=False)

    #create horizontal bar chart
    sns.set(style='white')
    plt.subplots(figsize=(10, 10))
    ax = company_analysis.plot(kind='barh', stacked=True )
    plt.axvline(avg_dissatisfaction*100, color='r')
    plt.axvline(0, color='grey')
    ax.bar_label(ax.containers[0], label_type='center', fmt='%.0f%%', padding=3, size=6)
    ax.bar_label(ax.containers[1], label_type='center', fmt='%.0f%%', padding=3, size=6)
    ax.bar_label(ax.containers[2], label_type='center', fmt='%.0f%%', padding=3, size=6)
    # ax.bar_label(ax.containers[3], label_type='center', fmt='%.0f%%', padding=3, size=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def write_response_ratio_plot(df_incidents, output_file):
    """ Create horizontal barchart with a comparison of survey response rates per given index_group and values
    Input:  dataframe with incident tickets
            output_file: file to be created
    Returns: None
    """
    avg_response_ratio = df_incidents['user_responded'].mean()*100

    crosstab = pd.DataFrame()

    for factor in ["reopened","days_to_resolve","no resolution"]:
            crosstab_reopened= pd.crosstab(
                index=df_incidents[factor],
                columns=df_incidents["user_responded"],
                normalize = 'index')
            crosstab_reopened["factor"]=factor
            crosstab_reopened["response_ratio"]=crosstab_reopened[1]
            crosstab_reopened.reset_index(inplace=True)
            crosstab_reopened["value"]=crosstab_reopened[factor]
            crosstab = pd.concat([crosstab, crosstab_reopened], ignore_index=True)

    # crosstab = pd.concat([crosstab_reopened, crosstab_days_to_resolve, crosstab_no_resolution], ignore_index=True)
    crosstab = crosstab[['factor','value','response_ratio']].copy()
    crosstab['factor_value'] = crosstab['factor'] + ": " + crosstab['value'].astype(str)
    crosstab['response_ratio'] = crosstab['response_ratio'].mul(100).round(1)

    #create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(x=crosstab.response_ratio, y=crosstab.factor_value, orient='h')
    plt.axvline(avg_response_ratio, color='g')
    plt.title('Survey Reponse Ratio - correlation with factor-value combinations')
    plt.xlabel('Response %')
    plt.ylabel('Factor + Value')
    ax.bar_label(ax.containers[0], fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)


