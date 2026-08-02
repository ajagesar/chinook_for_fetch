from utils.helpers import ROOT, config
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def main():

    # load data & features
    preprocessed_data = pd.read_csv(ROOT + 'data/intermediate/preprocessed_presplit.csv')
    with open(ROOT + 'data/intermediate/features.yaml', 'r') as f:
        features_list = yaml.safe_load(f)
        features = sorted(features_list)


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
        nice_aumc = pd.read_csv(ROOT + "data/intermediate/nice/aumc.csv")

        # # ensure matching dtypes for merge keys
        # nice_aumc['person_id'] = nice_aumc['person_id'].astype(int)
        # nice_aumc['visit_occurrence_id'] = nice_aumc['visit_occurrence_id'].astype(float)
        # preprocessed_data['person_id'] = preprocessed_data['person_id'].astype(int)
        # preprocessed_data['visit_occurrence_id'] = preprocessed_data['visit_occurrence_id'].astype(float)

        # convert visit_start_datetime to visit_start_date for better merging
        # Convert to date only and rename the column
        preprocessed_data['visit_start_date'] = pd.to_datetime(preprocessed_data['visit_start_datetime']).dt.date
        nice_aumc['visit_start_date'] = pd.to_datetime(nice_aumc['visit_start_datetime']).dt.date

        # Drop the old column
        preprocessed_data = preprocessed_data.drop('visit_start_datetime', axis=1)
        nice_aumc = nice_aumc.drop('visit_start_datetime', axis=1)


        # merge to keep only matching rows in the same order as nice_aumc # TODO check if all nice data rows is getting merge
        test_data = nice_aumc[['person_id', 'visit_start_date', 'nice_ap4_model', 'nice_ap4_prob']].merge(
            preprocessed_data,
            on=['person_id', 'visit_start_date'],
            how='inner'
        )
        

        # create train set from rows not in nice_aumc
        test_mask = preprocessed_data.set_index(['person_id', 'visit_start_date']).index.isin(
            nice_aumc.set_index(['person_id', 'visit_start_date']).index
        )
        train_data = preprocessed_data[~test_mask]

        # remove 'Amsterdam UMC' from train set
        train_data = train_data.loc[train_data['care_site_name'] != "Amsterdam UMC"]

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
        test_data.to_csv(ROOT + "data/intermediate/test_data_complete.csv", index=False)
        train_data.to_csv(ROOT + "data/intermediate/train_data_complete.csv", index=False)


        # also keep stratifies on cabg model and non cabg model
        X_test_noncabg = test_data.loc[test_data['nice_ap4_model']=="NON CABG"][features]
        y_test_noncabg = test_data.loc[test_data['nice_ap4_model']=="NON CABG"]['in_icu_death']

        X_test_cabg = test_data.loc[test_data['nice_ap4_model']=="CABG"][features]
        y_test_cabg = test_data.loc[test_data['nice_ap4_model']=="CABG"]['in_icu_death']

        X_test_noncabg.to_csv(ROOT + 'data/intermediate/X_test_noncabg.csv', index=False)
        X_test_cabg.to_csv(ROOT + 'data/intermediate/X_test_cabg.csv', index=False)
        y_test_noncabg.to_csv(ROOT + 'data/intermediate/y_test_noncabg.csv', index=False)
        y_test_cabg.to_csv(ROOT + 'data/intermediate/y_test_cabg.csv', index=False)



if __name__=="__main__":
    main()