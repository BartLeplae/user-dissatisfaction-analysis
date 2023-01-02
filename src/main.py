import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
import argparse
import time

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

    project_path = get_project_root()
    data_dir = project_path / "data"
    output_dir = project_path / "out"

    incident_data_path = data_dir / f"{args.incidents_fname}.xlsx"
    
    if (args.db):
        print("Read incidents from database")
    else:
        print("Read incidents from ", incident_data_path)
    
