import pandas as pd
from .helpers import ROOT, config, apache_config
import numpy as np
import matplotlib.pyplot as plt
import yaml


def calculate_apache_mort_score(df):


    # rename columns if necessary
    df = rename_for_apache(df=df)

    # get origin options (floor, ICU or OR/PACU)
    df = get_admission_origin(df=df, config=config)

    # check missing variables
    missing_apache_variables = check_apache_elements(df=df, config=config)

    # impute missing variables
    df = impute_apache_elements(df=df, missing_apache_variables=missing_apache_variables, config=config)

    # map apache diagnosis to corresonding codes
    df = map_apache_diagnosis(df=df, apache_config=apache_config)

    # calculate APS score
    df = calculate_aps_score(df=df, apache_config=apache_config)

    # Code for calculating mortality based on CABG and non-CABG diagnosis



    return df

def rename_for_apache(df):

    if 'in_icu_death' in df.columns:
        df = df.rename(columns={
            'in_icu_death': 'mortality'})
    
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

    # impute transfer as from floor
    df['admission_origin'] = df['admission_origin'].fillna('floor')

    # impute features
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