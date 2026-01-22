from utils.helpers import ROOT, config
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def main():

    # load data & features
    preprocessed_data = pd.read_csv(ROOT + 'data/intermediate/preprocessed_presplit.csv')
    with open(ROOT + 'data/intermediate/features.yaml', 'r') as f:
        features = yaml.safe_load(f)

    if config['splitting_method']['sklearn'] == True:

        X = preprocessed_data[features]
        y = preprocessed_data['in_icu_death']

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # also split in validation cohort if configured
        if config['use_validation_set'] is True:
            X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            X_validate.to_csv(ROOT + 'data/intermediate/X_validate.csv', index=False)
            y_validate.to_csv(ROOT + 'data/intermediate/y_validate.csv', index=False)

        X_train.to_csv(ROOT + 'data/intermediate/X_train.csv', index=False)
        X_test.to_csv(ROOT + 'data/intermediate/X_test.csv', index=False)
        y_train.to_csv(ROOT + 'data/intermediate/y_train.csv', index=False)
        y_test.to_csv(ROOT + 'data/intermediate/y_test.csv', index=False)


        return


    if config['splitting_method']['nice'] == True:

        # read nice data
        nice_aumc = pd.read_csv(ROOT + "intermediate/nice/aumc.csv")

        # merge to keep only matching rows in the same order as nice_aumc
        test_data = nice_aumc[['person_id', 'visit_occurrence_id']].merge(
            preprocessed_data,
            on=['person_id', 'visit_occurrence_id'],
            how='inner'
        )
        
        # create train set from rows not in nice_aumc
        test_mask = preprocessed_data.set_index(['person_id', 'visit_occurrence_id']).index.isin(
            nice_aumc.set_index(['person_id', 'visit_occurrence_id']).index
        )
        train_data = preprocessed_data[~test_mask]
        
        # separate features and target
        X_train = train_data[features]
        y_train = train_data['in_icu_death']
        X_test = test_data[features]
        y_test = test_data['in_icu_death']
        
        # save the splits
        X_train.to_csv(ROOT + 'data/intermediate/X_train.csv', index=False)
        X_test.to_csv(ROOT + 'data/intermediate/X_test.csv', index=False)
        y_train.to_csv(ROOT + 'data/intermediate/y_train.csv', index=False)
        y_test.to_csv(ROOT + 'data/intermediate/y_test.csv', index=False)

        return



if __name__=="__main__":
    main()