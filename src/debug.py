import pandas as pd
import numpy as np
from utils.helpers import ROOT, config
import pickle
import yaml

def main():


    return


def check_measurements_presplit(measurements):

    train = pd.read_csv(ROOT + "data/intermediate/train_data_complete.csv")
    test = pd.read_csv(ROOT + "data/intermediate/test_data_complete.csv")
    all = pd.concat([train, test], ignore_index=True)

    # meas_test = measurements.loc[measurements['person_id'].isin(test['person_id'])]

    meas_all = measurements.merge(all[['person_id', 'care_site_name']], on='person_id', how='left')

    # Group by target column (e.g., 'Category') and calculate missing percentage
    missing_by_group = (
        meas_all.groupby('Category')
        .apply(lambda group: group.isnull().mean() * 100)
        .T  # Transpose to put features as rows
        .round(2)
    )

    print(missing_by_group)

    return


if __name__=="__main__":
    main()