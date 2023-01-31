# user-dissatisfaction-analysis
Analyze the reasons why users respond 'dissatisfied' in incident resolution surveys

## Installation
https://github.com/BartLeplae/user-dissatisfaction-analysis

## Project motivation
For around 10% of the incident tickets, users enter a satisfaction survey.
For about 10% of the surveys, the users provide a 'dissatisfied' score

Questions:
1. Can the 'dissatisfied' responses be correlated with specific ticket attributes?
2. Can the ratio of 'dissatisfied' responses be predicted (modelled) based on specific attributes?
3. What is the predicted satisfaction ratio for tickets that don't have a survey response?

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
to run from the csv files in the data folder: python main.py
to run by obtaining the data from the data look: python main.py -d

## Remarks

## Licensing, Authors, Acknowledgements
Author: Bart Leplae
