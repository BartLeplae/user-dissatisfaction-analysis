# user-dissatisfaction-analysis
Analyze the reasons why users provided a 'dissatisfied' response in incident resolution surveys

## Project motivation
Given a dataset of around 116000 incident tickets:
- For around 10% of the incident tickets, users have provided a satisfaction survey response
- For about 10% of the surveys, the users provided a 'dissatisfied' score

Questions:
1. Can the 'dissatisfied' survey responses be correlated with specific ticket attributes?
2. Can the ratio of 'dissatisfied' responses for given subsets (e.g. those handled by a given support group) be predicted (modelled) based on specific attributes?
3. What is the predicted satisfaction ratio for tickets that don't have a survey response?

## Installation
https://github.com/BartLeplae/user-dissatisfaction-analysis
Libraries used:
- pandas, numpy
- sys, pathlib.Path, argparse, random
- sklearn.tree, sklearn.model_selection.GridSearchCV, sklearn.metrics.make_scorer
- matplotlib.pyplot, seaborn
- scipy.stats, stats.chi2_stats
- pyodbc

While the program extracts the incident data from a datalake when provided the -d argument, the default behavior is to utilize the csv files located in the data folder.
The applications, groups and companies are anonimized when extracted from the datalake for reasons of privacy.

## File Descriptions
Folders:
- data: input files (created through database queries in incidents_from_odbc.py)
    - incident_tickets.csv: incident tickets with survey results
    - all_incidents.csv: incident tickes with an without survey results
- docs:
    - Incident dissatisfaction analysis.docx: walkthrough through the analysis results
- out: resulting .xlsx and .png files, mostly created through output.py
- src:
    - main.py
    - model.py: creates model to predict user dissatisfaction
    - output.py: create .xls and .png files to depict the relationships
    - incidents_from_odbc.py: loads incident files from datalake through SQL statements
    - transform_attributes.py: transforms the incident data to enable analysis, modeling and reporting
    - stats.py: apply regular statistics on the incident data

## Technical details
- The model to predict the dissatisfaction% is based on a DecisionTreeClassifier from which the probability is used
  In other words, the model doesn't predict wether individual tickets will be flagged as satisfied or dissatisfied 
  but instead calculates the probability for a dissatisfied score
- GridSearchCV in combination with customer scorer function to avoid model overfitting
- Hyperparameters: 'max_depth' (5..10), 'min_samples_leaf' (50..130), 'criterion' ("gini","entropy")
- Custom scorer function: ensure dissatisfied% is correct over a wide range of dissatisfaction scores

## Instructions
- to run from the csv files in the data folder: python main.py
- to run by obtaining the data from the data look: python main.py -d

## Review of analysis - output
BartLeplae/user-dissatisfaction-analysis/docs/Incident dissatisfaction analysis.docx 

## Potential Improvements
- Create trend report that highlights 'abnormal' increases in customer dissatisfaction: this enables immediate actions
- Create report that shows more of the underpinning attributes for applications with high dissatisfied%

## Licensing, Authors, Acknowledgements
Author: Bart Leplae
