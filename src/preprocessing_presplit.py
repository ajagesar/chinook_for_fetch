import pandas as pd
from utils.helpers import ROOT, config, apache_config
from utils.apache_iv_mort_calc import calculate_apache_mort_score
import numpy as np
import matplotlib.pyplot as plt
import yaml

from debug import check_measurements_presplit

def main():

    ### --- general preprocessing --- ###

    # read data
    admissions, mortality, height, weight, measurements = load_data()

    # remove CTC patients if configured as such
    if config['drop_ctc'] is True:
        admissions = drop_ctc(admissions)

    admissions = preprocess_admissions(admissions)
    measurements = convert_to_mmhg(measurements, variable_name='variable_name', value='value_as_number')
    measurements = cleanup_measurements(measurements, variable_name='variable_name', value='value_as_number')
    measurements = get_most_deviating_values_apache(measurements)
    measurements = get_most_deviating_values_non_apache(measurements)
    measurements = pivot_measurements(measurements)

    # ### debug script start ###
    # check_measurements_presplit(measurements=measurements)
    # ### debug script end ###

    weight = preprocess_weight(weight)
    height = preprocess_height(height)
    measurements = preprocess_ventilated(measurements)
    mortality = preprocess_mortality(admissions, mortality)
    table = merge_all(admissions, mortality, height, weight, measurements)
    # TODO add one-hot encoding
    get_missing_statistics(table)
    get_patient_characteristics(table)
    table = impute_with_normal_values(table)
    table = calculate_derivations(table)

    table.to_csv(ROOT + 'data/intermediate/preprocessed_presplit_pre-apache.csv') # intermediate save

    print("")
    print("preprocessed data has been saved (without APACHE calculations)")
    print("")
    

    ### --- APACHE preprocessing --- ###

    # preprocessed_data = pd.read_csv(ROOT + 'data/intermediate/preprocessed_presplit_pre-apache.csv')
    # preprocessed_data = rename_for_apache(df=preprocessed_data)
    # preprocessed_data = get_admission_origin(df=preprocessed_data, config=config)
    # missing_apache_variables = check_apache_elements(df=preprocessed_data, config=config)
    # preprocessed_data = impute_apache_elements(df=preprocessed_data, missing_apache_variables=missing_apache_variables, config=config)
    # check_apache_elements(df=preprocessed_data, config=config)
    # df = map_apache_diagnosis(df=preprocessed_data, apache_config=apache_config)

    # # apache calculation
    # df = calculate_aps_score(df=preprocessed_data, apache_config=apache_config)
    
    # df['apache_hos_mort_prob'] = calculate_hos_mort_non_cabg(df, apache_config=apache_config, result='probability')
    # #df['apache_hos_mort_logodds'] = calculate_hos_mort_non_cabg(df, apache_config=apache_config, result='logodds')
    # #df['apache_hos_mort_prob_calibrated'] = calculate_hos_mort_non_cabg_recalibrated(df, logodds='apache_hos_mort_logodds', mortality='mortality')

    # df['apache_hos_mort_prob_cabg'] = calculate_hos_mort_cabg(df, apache_config=apache_config)

    # # rename back for further pipeline
    # df = rename_back(df=df)

    # load data for apache calculation
    df = pd.read_csv(ROOT + 'data/intermediate/preprocessed_presplit_pre-apache.csv')

    # # calculate apache mortality scores -> for now commented out
    # df = calculate_apache_mort_score(df=df)

    # save results
    df.to_csv(ROOT + 'data/intermediate/preprocessed_presplit.csv')

    print("")
    print("preprocessed data has been saved (including APACHE calculations)")
    print("")


    ### --- remove features not selected in config --- ###

    features = select_features()

    # save selected features
    with open(ROOT + 'data/intermediate/features.yaml', 'w') as f:
        yaml.dump(features, f)


def load_data():
    admissions = pd.read_csv(ROOT + 'data/raw/admissions.csv')
    mortality = pd.read_csv(ROOT + 'data/raw/mortality.csv')
    height = pd.read_csv(ROOT + 'data/raw/height.csv')
    weight = pd.read_csv(ROOT + 'data/raw/weight.csv')
    # ventilated = pd.read_csv(ROOT + 'data/raw/ventilated.csv')
    measurements = pd.read_csv(ROOT + 'data/raw/measurements.csv')

    return admissions, mortality, height, weight, measurements


def drop_ctc(admissions):

    admissions = admissions[~admissions['admitted_from_source_value'].str.contains("CCH|Cardiochi|CARDIOTHOR", na=False, case=False)]

    return admissions


def preprocess_admissions(admissions):

    # fix gender
    admissions.loc[(admissions['gender_concept_id'] == 8507) | (admissions['gender_concept_id'] == 1), 'gender'] = 1 # male
    admissions.loc[(admissions['gender_concept_id'] == 8532) | (admissions['gender_concept_id'] == 2), 'gender'] = 0 # female

    admissions = admissions.drop(columns=['gender_concept_id'])

    # fix admission floor
    # go through all origin options and assign them to df
    for origin_option, specific_department in config['features']['static']['admission_origin']['options'].items():
        admissions.loc[admissions['admitted_from_source_value'].isin(specific_department), 'admission_origin'] = origin_option

    # one-hot encode admission_origin (or_pacu / transfer / floor)
    admissions['admission_origin_or_pacu'] = (admissions['admission_origin'] == 'or_pacu').astype(int)
    admissions['admission_origin_transfer'] = (admissions['admission_origin'] == 'transfer').astype(int)
    admissions['admission_origin_floor'] = (admissions['admission_origin'] == 'floor').astype(int)
    admissions['admission_origin_ctc'] = (admissions['admission_origin'] == 'ctc').astype(int)

    # # get ctc_floor
    # admissions['ctc'] = 0
    # admissions.loc[admissions['admission_origin']=='ctc_floor', 'ctc'] = 1

    # drop admission_origin
    admissions = admissions.drop(columns=['admission_origin'])

    return admissions


def convert_to_mmhg(measurements, variable_name, value):

    # convert kpa pao2 (defined as values <= 20) to mmhg by multiplying with factor 7.5
    measurements.loc[(measurements[variable_name] == 'pao2') & (measurements[value] <= 20), value] = measurements.loc[measurements[variable_name] == 'pao2'][value] * 7.5

    # convert kpa pco2 (defined as values <= 20) to mmhg by multiplying with factor 7.5
    measurements.loc[(measurements[variable_name] == 'paco2') & (measurements[value] <= 20), value] = measurements.loc[measurements[variable_name] == 'paco2'][value] * 7.5

    return measurements


def cleanup_measurements(measurements, variable_name='variable_name', value='value_as_number'): 
    
    # load features config info
    features = config['features']

    # create plots for all variables before cleanup
    ## loop over all features in config file
    for feature_type, feature in features.items():
        for feature_name in feature.keys():

            ## extract values of features
            values = measurements.loc[measurements[variable_name] == feature_name][value]

            ## check if value series is empty, if not the case, then create and save plot
            if values.empty is False:

                ## create and save plot
                plt.hist(values)
                plt.savefig(ROOT + f'data/intermediate/exploration/{feature_name}_precleanup_hist.png')
                plt.cla()

    # apply specific preprocessing for certain variables
    ## hematocrit: convert all values below 1 (which are assumed to be fractions) to percentages by multiplying with 100
    measurements.loc[(measurements[variable_name] == 'hematocrit') & (measurements[value] < 1), value] = measurements.loc[(measurements[variable_name] == 'hematocrit') & (measurements[value] < 1)][value] * 100

    ## white blood cell count: assume all values above 70 are a typo (if entered by hand if feasible to check for that) and divide by 10 to get a more logical value for wbc
    measurements.loc[(measurements[variable_name] == 'wbc') & (measurements[value] > 70), value] = measurements.loc[(measurements[variable_name] == 'wbc') * (measurements[value] > 70)][value] / 10

    ## TODO split blood pressure to diastolic and systolic when combined in one variable
    

    # loop over all features and drop values as specified in 'min_max_cutoff'
    for feature_type, feature in features.items(): # loops over dictionary and returns tuple
        #feature = dict([feature]) # converts tuple back to dictionary for easier coding

        if feature_type == 'static':

            # here can cleanup be done for static features if specified as such in config file. Currently not the case.

            pass

        if feature_type == 'dynamic': # process as below for dynamic measurements

            for feature_name, feature_config in feature.items(): # loop over all dynamic features

                if feature_config['min_max_cutoff'] is not None: # apply cutoffs if they are specified in the config file

                    lower_cutoff = feature_config['min_max_cutoff'][0] # get first value in list of cutoff
                    upper_cutoff = feature_config['min_max_cutoff'][1] # get second value in list of cutoff

                    measurements = measurements.drop(measurements.loc[(measurements[variable_name] == feature_name) & (measurements[value] < lower_cutoff)].index)
                    measurements = measurements.drop(measurements.loc[(measurements[variable_name] == feature_name) & (measurements[value] > upper_cutoff)].index)
    
    # create plots for all variables after cleanup
    ## loop over all features in config file
    for feature_type, feature in features.items():
        for feature_name in feature.keys():

            ## extract values of features
            values = measurements.loc[measurements[variable_name] == feature_name][value]

            ## check if value series is empty, if not the case, then create and save plot
            if values.empty is False:

                ## create and save plot
                plt.hist(values)
                plt.savefig(ROOT + f'data/intermediate/exploration/{feature_name}_postcleanup_hist.png')
                plt.cla()

    return measurements


def get_most_deviating_values_apache(measurements):
    
    # load reference values from apache config
    aps_reference_value = apache_config['aps_reference_value']

    # load conversion factors from apache config
    conversion_factors = apache_config['conversion_factors']

    # convert reference values to mmol/L (or g/L for albumin). Loop over reference values and corresponding conversion factors to apply conversion.
    ## apply conversion if unit is mg/dL or g/dL
    for variable, value_unit in aps_reference_value.items():
        if value_unit['unit'] == 'mg/dl' or value_unit['unit'] == 'g/dl':
            aps_reference_value[variable]['value'] = aps_reference_value[variable]['value'] * conversion_factors[variable]
            aps_reference_value[variable]['unit'] = 'mmol/l'
    
    # keep most deviating value per variable per admission (person_id/visit_start_datetime combination) based on reference list from apache_config
    ## declare column with difference between measured variables and reference values
    measurements['ref_delta'] = np.nan
    for variable, value_unit in aps_reference_value.items():
        measurements.loc[measurements['variable_name'] == variable, 'ref_delta'] = abs(measurements.loc[measurements['variable_name'] == variable]['value_as_number'] - aps_reference_value[variable]['value'])

    # order (in descending fashion) by person_id/visit_start_datetime (so effectively per admission), variable and difference with reference values. Then drop the duplicates so in the end only value with the most deviating values with respect to the reference values remains.
    ## first split dataframe on values with a ref_delta value and those without for easier processing
    measurements_ref_delta = measurements.loc[measurements['ref_delta'].notna()]
    measurements_other = measurements.loc[~measurements['ref_delta'].notna()]

    ## apply sorting and only keeping the one with the highest deviating value
    measurements_ref_delta = measurements_ref_delta.sort_values(by=['person_id', 'visit_start_datetime', 'variable_name', 'ref_delta'], ascending=False).drop_duplicates(['person_id', 'visit_start_datetime', 'variable_name'])

    # concat both dataframes to get the original measurements dataframe
    measurements = pd.concat([measurements_ref_delta, measurements_other])

    # delete added ref_delta column
    measurements = measurements.drop(columns=['ref_delta'])

    # measurements = calculate_aps_score(measurements) -> moved to main function

    return measurements


def get_most_deviating_values_non_apache(measurements):

    # declare column with difference between measured variables and reference values
    measurements['ref_delta'] = np.nan

    # get feature configs
    features = config['features']

    # loop over variables and retrieve those do not have an apache reference value
    for feature_type, feature in features.items(): # loops over dictionary and returns tuple

        if feature_type == 'static':

            # code can be put here for processing static features. Currently not required.

            pass
        
        if feature_type == 'dynamic':

            for feature_name, feature_config in feature.items(): # loop over all dynamic features
            
                # if there is (normal) reference value, but not an apache reference value, calculate the deviation
                if (feature_config['ref_value_apache'] is None) and (feature_config['ref_value'] is not None): 

                    # get reference value (non_apache)
                    ref_value = feature_config['ref_value']

                    # calculate deviation with respect to reference value for non-apache variables
                    measurements.loc[measurements['variable_name'] == feature_name, 'ref_delta'] = abs(measurements.loc[measurements['variable_name'] == feature_name]['value_as_number'] - ref_value)

    # order (in descending fashion) by person_id/visit_start_datetime (so effectively per admission), variable and difference with reference values. Then drop the duplicates so in the end only value with the most deviating values with respect to the reference values remains.
    # ## first split dataframe on values with a ref_delta value and those without for easier processing
    measurements_ref_delta = measurements.loc[measurements['ref_delta'].notna()]
    measurements_other = measurements.loc[~measurements['ref_delta'].notna()]

    ## apply sorting and only keeping the one with the highest deviating value
    measurements_ref_delta = measurements_ref_delta.sort_values(by=['person_id', 'visit_start_datetime', 'variable_name', 'ref_delta'], ascending=False).drop_duplicates(['person_id', 'visit_start_datetime', 'variable_name'])

    # concat both dataframes to get the original measurements dataframe
    measurements = pd.concat([measurements_ref_delta, measurements_other])

    # delete added ref_delta column
    measurements = measurements.drop(columns=['ref_delta'])

    return measurements


def pivot_measurements(measurements):

    # drop unnecessary column
    measurements = measurements.reset_index().drop(columns='Unnamed: 0')
    
    # pivot and reset index for flattening of dataframe
    measurements_pivoted = measurements.pivot(index='index', columns='variable_name', values=['value_as_number']).reset_index()
    measurements_pivoted.columns = measurements_pivoted.columns.to_series().str.join('_')

    # change index_ to index for easier merging
    measurements_pivoted = measurements_pivoted.rename(columns={'index_':'index'})
    
    # left merge with original measurements column to keep person_id/visit_start_datetime combination
    measurements_merged = pd.merge(left=measurements[['index', 'person_id', 'visit_start_datetime']], right=measurements_pivoted, how='left', on='index')
    columns_to_keep = measurements_merged.columns[3:] # select all columns except the first 3 (index, person_id and visit_start_datetime)
    measurements_merged = measurements_merged.groupby(['person_id', 'visit_start_datetime'])[columns_to_keep].mean().reset_index()

    # remove prefix
    measurements_merged.columns = measurements_merged.columns.str.removeprefix("value_as_number_")

    return measurements_merged


def preprocess_weight(weight):
    
    weight = weight.sort_values(by=['person_id', 'visit_start_datetime', 'timediff']).drop_duplicates(subset=['person_id', 'visit_start_datetime'])

    return weight


def preprocess_height(height):

    # convert all m to cm
    height = height.loc[height['value_as_number'] < 3, 'value_as_number'] * 100
    
    height = height.sort_values(by=['person_id', 'visit_start_datetime', 'timediff']).drop_duplicates(subset=['person_id', 'visit_start_datetime'])

    return height


def preprocess_ventilated(measurements):

    # convert value to bool (the registration of a value of the ventilator is enoug information to assume the patient is on a ventilator)
    measurements.loc[measurements['ventilated'].notna(), 'ventilated'] = True

    # when not registered, set as false
    measurements.loc[measurements['ventilated'].isna(), 'ventilated'] = False

    return measurements


def preprocess_mortality(admissions, mortality):

    merged = pd.merge(left=admissions, right=mortality, how='left', on='person_id')
    merged['in_icu_death'] = 0
    merged.loc[(merged['death_datetime'] >= merged['visit_start_datetime'])  & (merged['death_datetime'] <= merged['visit_end_datetime']), 'in_icu_death'] = 1
    merged = merged.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])

    return merged


def merge_all(admissions, mortality, height, weight, measurements):

    height = height.rename(columns={'value_as_number':'height'})
    weight = weight.rename(columns={'value_as_number':'weight'})

    merged = pd.merge(left=admissions, right=mortality[['person_id', 'visit_start_datetime', 'in_icu_death']], how='left', on=['person_id', 'visit_start_datetime'])
    merged = pd.merge(left=merged, right=height[['person_id', 'visit_start_datetime', 'height']], how='left', on=['person_id', 'visit_start_datetime'])
    merged = pd.merge(left=merged, right=weight[['person_id', 'visit_start_datetime', 'weight']], how='left', on=['person_id', 'visit_start_datetime'])
    #merged = pd.merge(left=merged, right=ventilated[['person_id', 'visit_start_datetime', 'ventilated']], how='left', on=['person_id', 'visit_start_datetime'])
    merged = pd.merge(left=merged, right=measurements, how='left', on=['person_id', 'visit_start_datetime'])
    merged = merged.drop(columns=['birth_datetime'])

    # columns_to_keep = merged.drop(columns=['person_id', 'visit_start_datetime']).columns
    # merged = merged.groupby(['person_id', 'visit_start_datetime'])[columns_to_keep].mean().reset_index()

    # when ventilated not registered, set as false
    merged.loc[merged['ventilated'].isna(), 'ventilated'] = False

    return merged



def get_missing_statistics(df):

    df = df.drop(columns=df.columns[0], axis=1)

    # change underscore to space
    df.columns = df.columns.str.replace('_',' ')

    # rename columns
    rename_dict = {
        'gender concept id':'gender',
        'visit start datetime':'admission date',
        'visit end datetime':'discharge date',
        'in icu death':'in ICU mortality'
    }

    df = df.rename(columns=rename_dict)
    
    # code below is wrong for calculating missing
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({
        'column_name':df.columns,
        'percent_missing': percent_missing
    })

    # Calculate missing statistics stratified by db origin
    stratified_missing = {}
    
    for db in df['care site name'].unique():
        db_subset = df[df['care site name'] == db]
        percent_missing_db = db_subset.isnull().sum() * 100 / len(db_subset)
        stratified_missing[db] = percent_missing_db.to_dict()
    
    # Create combined dataframe with all databases
    stratified_df = pd.DataFrame(stratified_missing).T
    stratified_df.index.name = 'database'
    stratified_df = stratified_df.reset_index()
    stratified_df = stratified_df.rename(columns={'index': 'db origin'})
    
    # Export overall missing to html
    missing_value_df.to_html(ROOT + 'data/intermediate/exploration/missing_values.html', index=False)
    
    # Export stratified missing to html
    stratified_df.to_html(ROOT + 'data/intermediate/exploration/missing_values_stratified.html', index=False)
    
    # Save overall as YAML
    missing_dict = percent_missing.to_dict()
    with open(ROOT + 'data/intermediate/exploration/missing_values.yaml', 'w') as f:
        yaml.dump(missing_dict, f, default_flow_style=False, sort_keys=False)
    
    # Save stratified as YAML
    with open(ROOT + 'data/intermediate/exploration/missing_values_stratified.yaml', 'w') as f:
        yaml.dump(stratified_missing, f, default_flow_style=False, sort_keys=False)

    return



def get_patient_characteristics(df):

    # TODO get table 1 / patient characteristics here

    pass


def impute_with_normal_values(df):
    
    # drop 'Unnamed: 0' column
    df = df.drop(columns=df.columns[0], axis=1) 

    # change underscore to space
    # df.columns = df.columns.str.replace('_',' ')

    # # rename columns
    # rename_dict = {
    #     'gender concept id':'gender',
    #     'visit start datetime':'admission date',
    #     'visit end datetime':'discharge date',
    #     'in icu death':'in ICU mortality'
    # }

    # df = df.rename(columns=rename_dict)

    features = config['features']

    # apply this imputation if configured as such
    if config['assume_normal_values_for_nan'] is True:
        
        for feature_type, feature in features.items():

            if feature_type == 'static':
                
                # imputation code for static features, currently not needed
                
                pass

            # loop through settings of dynamic features
            if feature_type == 'dynamic':

                for feature_name, feature_config in feature.items():

                    # fill missing values with reference value if it is configured as such
                    if type(feature_config['ref_value']) == int or type(feature_config['ref_value']) == float:

                        ref_value = feature_config['ref_value']

                        # fill in dataframe if column exists in dataframe
                        if feature_name in df.columns:
                            df[feature_name] = df[feature_name].fillna(ref_value)

    return df


def calculate_derivations(df):
    
    # calculate AaDO2, calculation if FiO2 and PaCO2 are present in data. Assumptions: Patm = 760 mmHg, PH2O = 47 mmHg. https://en.wikipedia.org/wiki/Alveolar%E2%80%93arterial_gradient
    if ('fio2' in df.columns) and ('paco2' in df.columns) and ('pao2' in df.columns):
        
        # declare columns
        df['aado2'] = np.nan

        df['aado2'] = ((df['fio2'] / 100) * (760 - 47) - (df['paco2'] / 0.8)) - df['pao2']

    return df


def rename_for_apache(df):

    df = df.rename(columns={
        'in_icu_death':'mortality'})
    
    return df


def get_admission_origin(df, config):
    
    # go through all origin options and assign them to df
    for origin_option, specific_department in config['features']['static']['admission_origin']['options'].items():
        df.loc[df['admitted_from_source_value'].isin(specific_department), 'admission_origin'] = origin_option

    return df


def check_apache_elements(df, config):
    
    # check for non-CABG in hospital mortality prediction

    ## define list that contains missing variables
    missing_apache_variables = []

    ## retrieve apache variables
    apache_variables = []

    features = config['features']

    # go through config file features and append apache features to list
    for feature_type, feature in features.items():
        for feature_name, feature_config in feature.items():
            if feature_config['apache'] is True:
                apache_variables.append(feature_name)
    
    # retrieve which apache features are not in the preprocessed dataset
    missing_apache_variables = list(set(apache_variables) - set(df.columns.tolist()))

    print("")
    print(f"Missing variables in dataset to calculate APACHE IV: {missing_apache_variables}. Will be imputed.")
    print("")

    return missing_apache_variables


def impute_apache_elements(df, missing_apache_variables, config):

    features = config['features']

    # go through features in config file
    for feature_type, feature in features.items():
       for feature_name, feature_config in feature.items():
           
           # check if feature is not in the dataframe and part of the missing variables from apache
            if (feature_name not in df.columns.tolist()) and (feature_name in missing_apache_variables):

                # impute the entire missing feature based on the datatype
                if feature_config['dtype'] == 'bool':
                    df[feature_name] = False

                # impute float values with 0 (not ideal for features such as urine_output, but keeps the pipeline working)
                if feature_config['dtype'] == 'float':
                    df[feature_name] = 0.0
                
                # impute int values with 0
                if feature_config['dtype'] == 'int':
                    df[feature_name] = 0
                
                # the categorical values require more manual work:
                if feature_config['dtype'] == 'categorical':
                    
                    if feature_name == 'admission_origin':
                        df[feature_name] = 'floor' # reasonable assumption that a patient comes from the floor/ED if there is not more information known

                    if feature_name == 'admission_diagnosis':
                        df[feature_name] = 'Sepsis, GI' # arbitrary choice to keep the pipeline working

                    if feature_name == 'chronic_health_comorbidities':
                        df[feature_name] = 'cirrhosis' # least amount of points for comorbidities
        
    
    # feature specific changes
    ## fio2: set lowest oxygen fraction as 21%
    df['fio2'] = df['fio2'].clip(lower=0.21)

    # CABG specific imputations, will not be used by the non-CABG mortality prediction. Imputes entire entire columns or only missing variables
    if 'redo' in df.columns:
        df['redo'] = df['redo'].fillna(0) # automatically assume that we are not dealing with a redo CABG unless specified
    else:
        df['redo'] = 0

    if 'number_of_grafts' in df.columns:
        df['number_of_grafts'] = df['number_of_grafts'].fillna(3) # impute number of grafts to be 3 when not specified
    else:
        df['number_of_grafts'] = 3
    
    # return imputed df
    return df


def map_apache_diagnosis(df, apache_config):
    
    # create empty columns for diagnosis mappings
    df['admission_diagnosis_mapping'] = np.nan

    # TODO fix mapping from diagnosis to mapping
    for tractus, tractus_config in apache_config['disease_mapping'].items():
        for postsurg_status, mappings in tractus_config.items():
            for mapping in mappings: # because the mapping_keys are not unqiue in the config file, we have to go through them as a list rather than a dictionary
                for mapping_key, diagnosis in mapping.items():

                    # check if the admission diagnosis is part of a string, if so then assign the mapping_key to that row of the df
                    df.loc[df['admission_diagnosis'].apply(lambda x: x in diagnosis), 'admission_diagnosis_mapping'] = mapping_key

    return df


def calculate_aps_score(df, apache_config):
    
    # load aps points from config file
    aps_points = apache_config['aps_points']
    aps_reference_value = apache_config['aps_reference_value']
    conversion_factors = apache_config['conversion_factors']

    # apply conversion if unit is mg/dL or g/dL
    for variable, conversion_factor in conversion_factors.items():
        df[variable] = df[variable] / conversion_factor # go from mmol/L (or g/L for albumin) to g/dL

    # declare column for APS score
    df['APS_score'] = np.nan

    # declare temporary for APS sub score
    df['temp_points'] = np.nan

    # create new dataframe for APS points per variable
    df_aps = pd.DataFrame(
        data=df[['person_id', 'visit_start_datetime']].copy()
    )

    # declare list of aps variables for column names
    aps_variables = list(aps_points.keys())

    # add columns for aps points
    df_aps[aps_variables] = np.nan

    # assign APS points based on apache config file and measurements
    for variable, range in aps_points.items():

        # empty temporary for APS sub score. Needs to be at this place to not overwrite variables with more components such as oxygenation and creatinine
        df['temp_points'] = np.nan

        for range_number, range_component in range.items():          
            
            # specific processing for oxygenation variable (aado2 or pao2 to use)
            if variable == 'oxygenation':
                
                # extract upper and lower limit from apache config file. Assign positive or negative infinity in case there is no upper or lower limit.
                for range_component_sub, range_component_sub_config in range_component.items():

                    if range_component_sub_config['min'] is not None:
                        lower = range_component_sub_config['min']
                    else:
                        lower = float('-inf')
                    
                    if range_component_sub_config['max'] is not None:
                        upper = range_component_sub_config['max']
                    else:
                        upper = float('inf')

                    # extract corresponding APS points for variable
                    points = range_component_sub_config['points']

                    variable_sub = range_number

                    if variable_sub == 'aado2': # not actually range_number, but it works due to the way the config file is built

                        # assign APS points based on value
                        df.loc[(df[variable_sub] >= lower) & (df[variable_sub] < upper) & (df['ventilated'] == True) & (df['fio2'] > 0.5), 'temp_points'] = points
                    
                    if variable_sub == 'pao2': # not actually range_number, but it works due to the way the config file is built
                        
                        # assign APS points based on value
                        df.loc[(df[variable_sub] >= lower) & (df[variable_sub] < upper) & ((df['ventilated'] == False) | (df['fio2'] <= 0.5)), 'temp_points'] = points

                    df_aps[variable] = df['temp_points'].copy()

                    # # empty temporary for APS sub score
                    # df['temp_points'] = np.nan

            # specific processing for creatinine (depending on whether or not ARF is determined)
            elif variable == 'creatinine':

                # extract upper and lower limit from apache config file. Assign positive or negative infinity in case there is no upper or lower limit.
                for range_component_sub, range_component_sub_config in range_component.items():

                    if range_component_sub_config['min'] is not None:
                        lower = range_component_sub_config['min']
                    else:
                        lower = float('-inf')
                    
                    if range_component_sub_config['max'] is not None:
                        upper = range_component_sub_config['max']
                    else:
                        upper = float('inf')

                    # extract corresponding APS points for variable
                    points = range_component_sub_config['points']

                    # put ARF status as separate variable
                    arf_status = range_number

                    # assign points for patients with ARF
                    if arf_status == 'with_arf':
                        df.loc[(df['arf'] == True) & (df[variable] >= lower) & (df[variable] < upper), 'temp_points'] = points

                    # assign points for patients without ARF
                    if arf_status == 'without_arf':
                        df.loc[(df['arf'] == False) & (df[variable] >= lower) & (df[variable] < upper), 'temp_points'] = points

                    # store APS points in APS dataframe for easier summation later
                    df_aps[variable] = df['temp_points'].copy()

                    # # empty temporary for APS sub score
                    # df['temp_points'] = np.nan

            # specific processing for acid-base abnormalities (depending on combination of pH and PaO2)
            elif variable == 'acid_base':
                
                # extract upper and lower limit from apache config file. Assign positive or negative infinity in case there is no upper or lower limit.
                # for range_component_sub, range_component_sub_config in range_component.items():

                # get upper and lower for ph
                if range_component['ph']['min'] is not None:
                    lower_ph = range_component['ph']['min']
                else:
                    lower_ph = float('-inf')
                
                if range_component['ph']['max'] is not None:
                    upper_ph = range_component['ph']['max']
                else:
                    upper_ph = float('inf')

                # get upper and lower for paco2
                if range_component['paco2']['min'] is not None:
                    lower_paco2 = range_component['paco2']['min']
                else:
                    lower_paco2 = float('-inf')
                
                if range_component['paco2']['max'] is not None:
                    upper_paco2 = range_component['paco2']['max']
                else:
                    upper_paco2 = float('inf')

                # extract corresponding APS points for variable
                points = range_component['points']

                # assign them to df
                df.loc[(df['ph'] >= lower_ph) & (df['ph'] < upper_ph) & (df['paco2'] >= lower_paco2) & (df['paco2'] < upper_paco2), 'temp_points'] = points

                # store APS points in APS dataframe for easier summation later
                df_aps[variable] = df['temp_points'].copy()

            # specific processing for combination of Glasgow Coma Scale scores
            elif variable == 'neurological':
                
                # go through the nested loops in the config file
                for eye_score, motor_verbal_string_score_and_points in range_component.items():
                    for motor_string, motor_verbal_score_and_points in motor_verbal_string_score_and_points.items():
                        for motor_score, verbal_string_score_and_points in motor_verbal_score_and_points.items():
                            for verbal_string, verbal_score_and_points in verbal_string_score_and_points.items():
                                for verbal_score, points in verbal_score_and_points.items():

                                    eye_score = eye_score
                                    motor_score = motor_score
                                    verbal_score = verbal_score

                                    points = points['points']

                                    # calculate score for patients with eye_score = 1
                                    if eye_score == 1:
                                        df.loc[(df['gcs_eye'] == eye_score) & (df['gcs_motor'] == motor_score) & (df['gcs_verbal'] == verbal_score), 'temp_points'] = points

                                    # calculate score for patients with eye scores >= 2
                                    if eye_score == 2:
                                        df.loc[(df['gcs_eye'] >= eye_score) & (df['gcs_motor'] == motor_score) & (df['gcs_verbal'] == verbal_score), 'temp_points'] = points

                # store APS points in APS dataframe for easier summation later
                df_aps[variable] = df['temp_points'].copy()

            # specific processing for respiratory rate with and without ventilation
            elif variable == 'respiratory_rate':

                # extract upper and lower limit from apache config file. Assign positive or negative infinity in case there is no upper or lower limit.
                for range_component_sub, range_component_sub_config in range_component.items():

                    if range_component_sub_config['min'] is not None:
                        lower = range_component_sub_config['min']
                    else:
                        lower = float('-inf')
                    
                    if range_component_sub_config['max'] is not None:
                        upper = range_component_sub_config['max']
                    else:
                        upper = float('inf')
                    
                    # extract corresponding APS points for variable
                    points = range_component_sub_config['points']

                    ventilation_status = range_number

                    if ventilation_status == 'with_ventilation':
                        
                        df.loc[(df['ventilated'] == True) & (df[variable] >= lower) & (df[variable] < upper), 'temp_points'] = points

                    if ventilation_status == 'without_ventilation':
                        
                        df.loc[(df['ventilated'] == False) & (df[variable] >= lower) & (df[variable] < upper), 'temp_points'] = points
                    
                    # store APS points in APS dataframe for easier summation later
                    df_aps[variable] = df['temp_points'].copy()

            # general processing for variables
            else:

                # extract upper and lower limit from apache config file. Assign positive or negative infinity in case there is no upper or lower limit.
                if range_component['min'] is not None:
                    lower = range_component['min']
                else:
                    lower = float('-inf')
                
                if range_component['max'] is not None:
                    upper = range_component['max']
                else:
                    upper = float('inf')

                # extract corresponding APS points for variable
                points = range_component['points']
                
                # assign APS points based on value
                df.loc[(df[variable] >= lower) & (df[variable] < upper), 'temp_points'] = points

                # store APS points in APS dataframe for easier summation later
                df_aps[variable] = df['temp_points'].copy()

    
    # sum up all aps points
    df_aps['APS_score'] = df_aps[aps_variables].sum(axis=1)

    df['APS_score'] = df_aps['APS_score'].copy()

    df = df.drop(columns=['temp_points'])

    return df


def calculate_hos_mort_non_cabg(df, apache_config, result='probability'):
    
    # load coefficients
    coeffs = apache_config['mort_coeff']

    # the code below adds all the coefficients for in-hospital mortality for non-CABG patients

    # create new dataframe for all patients for all coefficients
    df_apache = pd.DataFrame(
        data=df[['person_id', 'visit_start_datetime']].copy()
    )

    # add intercept for all patients
    df_apache['intercept'] = coeffs['intercept']

    # calculate age coeff. Clip in the meantime to perform the cubic spline with knot point calculation

    age_age = df['age'] * coeffs['age']['age']['coeff']
    age_1 = ((df['age'] - coeffs['age']['age1']['knot']) ** 3).clip(lower=0) * coeffs['age']['age1']['coeff']
    age_2 = ((df['age'] - coeffs['age']['age2']['knot']) ** 3).clip(lower=0) * coeffs['age']['age2']['coeff']
    age_3 = ((df['age'] - coeffs['age']['age3']['knot']) ** 3).clip(lower=0) * coeffs['age']['age3']['coeff']
    age_4 = ((df['age'] - coeffs['age']['age4']['knot']) ** 3).clip(lower=0) * coeffs['age']['age4']['coeff']
    age_5 = ((df['age'] - coeffs['age']['age5']['knot']) ** 3).clip(lower=0) * coeffs['age']['age5']['coeff']

    df_apache['age'] = age_age + age_1 + age_2 + age_3 + age_4 + age_5 # sum all age cross products and add it to the apache table

    # calculate chronic health (order of importance: AIDS, hepatic failure, lymphoma, metastatic cancer, leukemia/MM, immunosuppression, cirrhosis, none/not available)
    df['temp_cross_prod'] = np.nan # create temporary cross product column in dataframe

    for comorbidity, coeff in coeffs['chronic_health'].items(): # loop through all chronich health coeffs
        df.loc[df['chronic_health_comorbidities'] == comorbidity, 'temp_cross_prod'] = coeff # note, only 1 coeff is possible for chronic health, based on the highest score in APACHE III calculation

    df_apache['chronic_health'] = df['temp_cross_prod'].copy() # put them in the apache df
    df = df.drop(columns=['temp_cross_prod']) # delete the temporary cross_prod column

    # calculate APS score
    aps_aps = df['APS_score'] * coeffs['aps']['aps']['coeff']
    aps_1 = ((df['APS_score'] - coeffs['aps']['aps1']['knot']).clip(lower=0) ** 3) * coeffs['aps']['aps1']['coeff']
    aps_2 = ((df['APS_score'] - coeffs['aps']['aps2']['knot']).clip(lower=0) ** 3) * coeffs['aps']['aps2']['coeff']
    aps_3 = ((df['APS_score'] - coeffs['aps']['aps3']['knot']).clip(lower=0) ** 3) * coeffs['aps']['aps3']['coeff']
    aps_4 = ((df['APS_score'] - coeffs['aps']['aps4']['knot']).clip(lower=0) ** 3) * coeffs['aps']['aps4']['coeff']
    aps_5 = ((df['APS_score'] - coeffs['aps']['aps5']['knot']).clip(lower=0) ** 3) * coeffs['aps']['aps5']['coeff']

    df_apache['aps'] = aps_aps + aps_1 + aps_2 + aps_3 + aps_4 + aps_5

    # calculate PF ratio cross prod
    df['pf_ratio'] = df['pao2'] / df['fio2'] # pao2 in mmHg and fio2 in fractions (min 0.21, max 1.0)
    df_apache['pf_ratio'] = df['pf_ratio'] * coeffs['pf_ratio']

    # calculate ventilation cross prod
    df_apache['ventilated'] = df['ventilated'] * coeffs['ventilated']

    # calculate admission origin cross prod
    df['temp_cross_prod'] = np.nan

    for origin, coeff in coeffs['admission_origin'].items():
        df.loc[df['admission_origin'] == origin, 'temp_cross_prod'] = coeff
    
    df_apache['admission_origin'] = df['temp_cross_prod'].copy()
    df = df.drop(columns=['temp_cross_prod'])

    # calculate emergency surgery cross prod
    df_apache['emergency_surgery'] = df['emergency_surgery'] * coeffs['emergency_surgery']

    # calculate preICU los cross prod
    square_root_of_preicu_los = np.sqrt(df['preicu_los_in_days']) 
    square_root_cross_prod = square_root_of_preicu_los * coeffs['preicu_los']['square_root_of_days']['coeff']
    prelos21_cross_prod = ((square_root_of_preicu_los - coeffs['preicu_los']['prelos21']['knot']).clip(lower=0) ** 3) * coeffs['preicu_los']['prelos21']['coeff']
    prelos22_cross_prod = ((square_root_of_preicu_los - coeffs['preicu_los']['prelos22']['knot']).clip(lower=0) ** 3) * coeffs['preicu_los']['prelos22']['coeff']
    prelos23_cross_prod = ((square_root_of_preicu_los - coeffs['preicu_los']['prelos23']['knot']).clip(lower=0) ** 3) * coeffs['preicu_los']['prelos23']['coeff']
    prelos24_cross_prod = ((square_root_of_preicu_los - coeffs['preicu_los']['prelos24']['knot']).clip(lower=0) ** 3) * coeffs['preicu_los']['prelos24']['coeff']

    df_apache['preicu_los'] = square_root_cross_prod + prelos21_cross_prod + prelos22_cross_prod + prelos23_cross_prod + prelos24_cross_prod

    # calculate admission diagnosis cross prod
    ## set temp_cross_prod column
    df['temp_cross_prod'] = np.nan

    ## loop through all mappings with diagnoses
    for mapping_key, coeff in coeffs['admission_diagnosis'].items():

        df.loc[df['admission_diagnosis_mapping'] == mapping_key, 'temp_cross_prod'] = coeff

    ## set apache dataframe with coeffs
    df_apache['admission_diagnosis'] = df['temp_cross_prod'].copy()
    df = df.drop(columns=['temp_cross_prod'])

    ## calculate cross_prod if thrombolytic therapy was used
    df['temp_cross_prod'] = 0 # for thrombolytic therapy
    df.loc[df['thrombolytic_therapy'] == True, 'temp_cross_prod'] = coeffs['thrombolytic_therapy']

    df_apache['thrombolytic_therapy'] = df['temp_cross_prod'].copy()

    df = df.drop(columns=['temp_cross_prod'])

    # get gcs score cross_prod
    ## code if one of the components is unavailable/nan
    df['temp_cross_prod'] = 0 # for availability to obtain gcs

    ## assign coeff of not being able to get glasgow coma scale score
    df.loc[(df['gcs_eye'].isna()) | (df['gcs_motor'].isna()) | (df['gcs_verbal'].isna()), 'temp_cross_prod'] = coeffs['unable_to_get_gcs']
    df_apache['unable_to_get_gcs'] = df['temp_cross_prod'].copy()
    df = df.drop(columns=['temp_cross_prod'])

    ## assign coeff when able to get glasgow coma scale score
    df['temp_cross_prod'] = np.nan
    df['gcs_total'] = df['gcs_eye'] + df['gcs_motor'] + df['gcs_verbal']
    df.loc[(df['gcs_eye'].notna()) & (df['gcs_motor'].notna()) & (df['gcs_verbal'].notna()), 'temp_cross_prod'] = (15 - df['gcs_total']) * coeffs['adjusted_gcs']

    df_apache['adjusted_gcs'] = df['temp_cross_prod'].copy()

    ## drop newly added columns for calculation
    df = df.drop(columns=['gcs_total', 'temp_cross_prod'])

    # fill nans in dataframe with 0 (similar to not writing a value in a spreadsheet)
    df_apache = df_apache.fillna(0)

    # calculate the sum of the cross products
    df_apache['sum_of_cross_products'] = df_apache['intercept'] + df_apache['age'] + df_apache['chronic_health'] + df_apache['aps'] + df_apache['pf_ratio'] + df_apache['ventilated'] + df_apache['admission_origin'] + df_apache['emergency_surgery'] + df_apache['preicu_los'] + df_apache['admission_diagnosis'] + df_apache['thrombolytic_therapy'] + df_apache['unable_to_get_gcs'] + df_apache['adjusted_gcs']

    # calculate the natural antilog
    df_apache['natural_antilog'] = np.exp(df_apache['sum_of_cross_products'])

    # calculate the probability based prediction
    df_apache['probability'] = df_apache['natural_antilog'] / (1 + df_apache['natural_antilog'])

    if result == 'probability':
        return df_apache['probability']
    
    if result == 'logodds':
        return df_apache['sum_of_cross_products'] # return the logodds


def calculate_hos_mort_non_cabg_recalibrated(df, logodds='logodds', mortality='mortality'):
    
    # apply statsmodels Logistic Regression implementation. Use logodds of original APACHE IV outcome as independent variable, and mortality as dependent variable

    pass


def calculate_hos_mort_cabg(df, apache_config):
    
    # load coefficients
    coeffs = apache_config['mort_coeff_cabg']

    # the code below adds all the coefficients for in-hospital mortality for CABG patients (all data input should be CABG patients only)
    
    # create new dataframe for all patients for all coefficients
    df_apache = pd.DataFrame(
        data=df[['person_id', 'visit_start_datetime']].copy()
    )

    # add intercept for all patients
    df_apache['intercept'] = coeffs['intercept']

    # find age range and use appropriate coeff
    df['temp_cross_prod'] = np.nan

    for range, bounds in coeffs['age'].items():
        
        # set lower bound if specified
        if bounds['lower'] is not None:
            lower = bounds['lower']
        else:
            lower = float('-inf')

        # set upper bound if specified
        if bounds['upper'] is not None:
            upper = bounds['upper']
        else:
            upper = float('inf')
    
        df.loc[(df['age'] >= lower) & (df['age'] < upper), 'temp_cross_prod'] = bounds['coeff']

    df_apache['age'] = df['temp_cross_prod'].copy()
    df = df.drop(columns=['temp_cross_prod'])

    # use APS score with splining for CABG
    aps_aps = df['APS_score'] * coeffs['aps']['aps']['coeff']

    aps_1 = (
        ((df['APS_score'] - coeffs['aps']['aps1']['knot']) ** 3).clip(lower=0) +
        1.5 * ((df['APS_score'] - 90) ** 3).clip(lower=0) -
        2.5 * ((df['APS_score'] - 60) ** 3).clip(lower=0)
    ) / 5625 * coeffs['aps']['aps1']['coeff']

    aps_2 = (
        ((df['APS_score'] - coeffs['aps']['aps2']['knot']) ** 3).clip(lower=0) +
        ((df['APS_score'] - 90) ** 3).clip(lower=0) -
        (2 * (df['APS_score'] - 60).clip(lower=0) ** 3)
    ) / 5625 * coeffs['aps']['aps2']['coeff']

    aps_3 = (
        ((df['APS_score'] - coeffs['aps']['aps3']['knot']) ** 3).clip(lower=0) +
        ((2/3) * ((df['APS_score'] - 90) ** 3).clip(lower=0)) -
        ((5/3) * ((df['APS_score'] - 60) ** 3).clip(lower=0))
    ) / 5625 * coeffs['aps']['aps3']['coeff']

    aps_4 = (
        ((df['APS_score'] - coeffs['aps']['aps4']['knot']) ** 3).clip(lower=0) +
        ((1/3) * ((df['APS_score'] - 90) ** 3).clip(lower=0)) -
        ((4/3) * ((df['APS_score'] - 60) ** 3).clip(lower=0))
    ) / 5625 * coeffs['aps']['aps4']['coeff']

    df_apache['aps'] = aps_aps + aps_1 + aps_2 + aps_3 + aps_4

    # calculate emergency surgery cross prod
    df_apache['emergency_surgery'] = df['emergency_surgery'] * coeffs['emergency_surgery']

    # gender cross prod
    df_apache['female'] = df['gender'] * coeffs['female'] # female gender already coded as 1 in preprocessing and male gender as 0

    # redo cabg cross prod
    df_apache['redo'] = df['redo'] * coeffs['redo']

    # number of grafts
    df_apache['number_of_grafts'] = df['number_of_grafts'] * coeffs['number_of_grafts']
    
    # sum of cross products
    df_apache['sum_of_cross_products'] = df_apache['intercept'] + df_apache['age'] + df_apache['aps'] + df_apache['emergency_surgery'] + df_apache['female'] + df_apache['redo'] + df_apache['number_of_grafts']

    # calculate the natural antilog
    df_apache['natural_antilog'] = np.exp(df_apache['sum_of_cross_products'])

    # calculate the probability based prediction
    df_apache['prediction'] = df_apache['natural_antilog'] / (1 + df_apache['natural_antilog'])

    return df_apache['prediction']

def rename_back(df):
    
    df = df.rename(columns={
        'mortality':'in_icu_death'
    })

    return df


def select_features():

    # load data
    preprocessed_data = pd.read_csv(ROOT + 'data/intermediate/preprocessed_presplit.csv')
    
    # select features which are 'turned on'
    config_of_features = config['features']
    
    features = [] # declare empty list to be filled with the features to be used in modeling
    
    for feature_type, feature in config_of_features.items():

            for feature_name, feature_config in feature.items():
                    
                    if feature_config['processed'] is True:
                        
                        features.append(feature_name)

    # print which features are not found in the dataframe
    not_in_data = set(features) - set(list(preprocessed_data.columns))

    print("")
    print(f"Features not in dataset: {not_in_data}")
    print("")

    # keep only features that are in the actual preprocessed date column names (features that are set as processed but not found will be ignored)
    features = set(features) & set(list(preprocessed_data.columns))
    features = list(features) # convert dictionary to list

    return features


if __name__=="__main__":
    main()