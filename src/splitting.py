from utils.helpers import ROOT, config
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def main():

    # load data & features
    preprocessed_data = pd.read_csv(ROOT + '/data/intermediate/preprocessed_presplit.csv')
    with open('features.yaml', 'r') as f:
        features = yaml.safe_load(f)

    X = preprocessed_data[features]
    y = preprocessed_data['in_icu_death']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # also split in validation cohort if configured
    if config['use_validation_set'] is True:
        X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        X_validate.to_csv(ROOT + '/data/intermediate/X_validate.csv', index=False)
        y_validate.to_csv(ROOT + '/data/intermediate/y_validate.csv', index=False)

    X_train.to_csv(ROOT + '/data/intermediate/X_train.csv', index=False)
    X_test.to_csv(ROOT + '/data/intermediate/X_test.csv', index=False)
    y_train.to_csv(ROOT + '/data/intermediate/y_train.csv', index=False)
    y_test.to_csv(ROOT + '/data/intermediate/y_test.csv', index=False)

    return


if __name__=="__main__":
    main()