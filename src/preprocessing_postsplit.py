from utils.helpers import ROOT, config, impute_dataframe
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from imblearn.over_sampling import SMOTE

def main():

    if config['use_validation_set'] == True:
        X_train, X_validate, X_test = load_data()

    else:
        X_train, X_test = load_data()

    if config['splitting_method']['nice'] == True:

        X_test_noncabg = pd.read_csv(ROOT + 'data/intermediate/X_test_noncabg.csv')

        X_test_cabg = pd.read_csv(ROOT + 'data/intermediate/X_test_cabg.csv')

        impute(X_train=X_train, X_test=X_test, X_test_cabg=X_test_cabg, X_test_noncabg=X_test_noncabg)

    else:
        impute(X_train=X_train, X_test=X_test)

    # TODO implement scaling if necessary

    if config['oversample'] == True:
        oversample()

    return


def load_data():


    # load data
    X_train = pd.read_csv(ROOT + 'data/intermediate/X_train.csv')
    X_test = pd.read_csv(ROOT + 'data/intermediate/X_test.csv')

    if config['use_validation_set'] == True:
        X_validate = pd.read_csv(ROOT + 'data/intermediate/X_validate.csv')
        return X_train, X_validate, X_test

    else:
        return X_train, X_test


def impute(X_train, X_test, X_test_cabg=None, X_test_noncabg=None):

    # impute data
    ## imputed whole data set with normal values. Now impute train set with multivariate imputation
    if (config['imputation_method']['median'] == True) and (config['imputation_method']['mice'] == True):
        print("Cannot execute imputation method, more than one method is specified as True")
        exit()

    elif config['imputation_method']['median'] == True:
        # function to impute with median

        ## fit imputer on train data
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        imputer.fit(X_train)

        ## impute train and test data
        X_train_imputed = impute_dataframe(imputer=imputer, df=X_train)
        X_test_imputed = impute_dataframe(imputer=imputer, df=X_test)

        # also impute stratification on CABG model of APACHE
        if X_test_cabg is not None:
            X_test_cabg = impute_dataframe(imputer=imputer, df=X_test_cabg)
        if X_test_noncabg is not None:
            X_test_noncabg = impute_dataframe(imputer=imputer, df=X_test_noncabg)

    elif config['imputation_method']['mice'] == True:
        # function to impute with mice

        ## fit imputer on train data
        imputer = IterativeImputer(max_iter=1000)

        imputer.fit(X_train)

        ## impute train and test data
        X_train_imputed = impute_dataframe(imputer=imputer, df=X_train)
        X_test_imputed = impute_dataframe(imputer=imputer, df=X_test)

        # also impute stratification on CABG model of APACHE
        if X_test_cabg is not None:
            X_test_cabg = impute_dataframe(imputer=imputer, df=X_test_cabg)
        if X_test_noncabg is not None:
            X_test_noncabg = impute_dataframe(imputer=imputer, df=X_test_noncabg)
            
    else:
        X_train_imputed = X_train.copy()
        X_test_imputed = X_test.copy()

    X_train_imputed.to_csv(ROOT + 'data/intermediate/X_train_imputed.csv', index=False)
    X_test_imputed.to_csv(ROOT + 'data/intermediate/X_test_imputed.csv', index=False)

    if X_test_cabg is not None:
        X_test_cabg.to_csv(ROOT + 'data/intermediate/X_test_cabg.csv', index=False)
    if X_test_noncabg is not None:
        X_test_noncabg.to_csv(ROOT + 'data/intermediate/X_test_noncabg.csv', index=False)


    return


def scale():


    return

def oversample():

    # set random state based on config file
    random_state = config['random_state']

    # load data
    X_train = pd.read_csv(ROOT + 'data/intermediate/X_train_imputed.csv')
    y_train = pd.read_csv(ROOT + 'data/intermediate/y_train.csv')

    # create smote model and apply it
    smote = SMOTE(random_state=random_state)

    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # export training data
    X_train_smote.to_csv(ROOT + 'data/intermediate/X_train_smote.csv', index=False)
    y_train_smote.to_csv(ROOT + 'data/intermediate/y_train_smote.csv', index=False)

    return



if __name__=="__main__":
    main()