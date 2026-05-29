from utils.helpers import ROOT, config
import duckdb
import pandas as pd
import yaml
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, xxhash64
import xxhash

# TODO: remove filename for ICUdata version 0.1.0

def main():

    conn = database_connect(path_to_duck_db=config['data']['icudata_0_1_0'])

    features = config['features']
    extraction_window = config['window_in_hours']

    if config['pipeline']['save_schema'] == True:
        save_schema(conn)

    mortality = get_mortality(conn, features)
    admissions = get_admissions(conn, features)
    measurements = get_measurements(conn, features, extraction_window)
    weight = get_weight(conn, features)
    height = get_height(conn, features)
    ventilated = get_ventilated(conn, features, extraction_window)

    close_connection(conn=conn)

    nice_aumc = get_nice_aumc()

    return


def save_schema(conn):

    # Get all tables and columns
    result = conn.execute(f"""
        SELECT 
            table_name,
            column_name,
            data_type,
            is_nullable
        FROM information_schema.columns
        ORDER BY table_name, ordinal_position
    """).fetchdf()

    # Organize data by table
    schema = {}
    for table_name in result['table_name'].unique():
        table_cols = result[result['table_name'] == table_name]
        schema[table_name] = []
        
        for _, row in table_cols.iterrows():
            schema[table_name].append({
                'column_name': row['column_name'],
                'data_type': row['data_type'],
                'is_nullable': row['is_nullable']
            })

    # Write to YAML file
    with open(ROOT + 'data/raw/schema.yaml', 'w') as f:
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    print("Schema exported to schema.yaml")

    return


def get_mortality(conn, features):

        # check if mortality is being set as true for extraction
    if features['static']['mortality']['extract'] is True:

        sql_death = f"""
            SELECT 
                person_id,
                death_datetime
            FROM omop.death
        """

        sql_mortuary = f"""
            SELECT 
                person_id,
                -- visit_start_datetime,
                visit_end_datetime AS death_datetime
            FROM omop.visit_occurrence
            WHERE LOWER(discharged_to_source_value) LIKE '%mortuarium%'
        """

        ## death preprocessing metavision VUMC
        #death_mv = pd.read_sql(sql_death, con=conn_mv)
        #mortuary_mv = pd.read_sql(sql_mortuary, con=conn_mv)

        ## death preprocessing epic AUMC
        #death_epic = pd.read_sql(sql_death, con=conn_epic)
        #mortuary_epic = pd.read_sql(sql_mortuary, con=conn_epic)

        ## concat all and drop missing data and duplicates while keeping first registration of death/going to mortuary
        #mortality = pd.concat([death_mv, mortuary_mv, death_epic, mortuary_epic])
        #mortality = mortality.dropna(subset='death_datetime')
        #mortality = mortality.sort_values(by=['person_id', 'death_datetime']).drop_duplicates(subset=['person_id'], keep='first')

        death = pd.read_sql(sql=sql_death, con=conn)
        mortuary = pd.read_sql(sql=sql_mortuary, con=conn)
        mortality = pd.concat([death, mortuary])
        mortality = mortality.dropna(subset='death_datetime')
        mortality = mortality.sort_values(by=['person_id', 'death_datetime']).drop_duplicates(subset=['person_id'], keep='first')

        mortality.to_csv(ROOT + '/data/raw/mortality.csv')
        print("")
        print("mortality csv has been saved")
        print("")
        return mortality
    
    else:
        return "extraction of mortality (outcome) is set as False, stopped extraction"


def get_admissions(conn, features):
    
    sql_admission_w_care_site = f"""
    SELECT 
        person.person_id,
        person.gender_concept_id,
        person.birth_datetime,
        visit_occurrence.visit_occurrence_id,
        visit_occurrence.visit_start_datetime,
        visit_occurrence.visit_end_datetime,
        visit_occurrence.admitted_from_source_value,
        visit_occurrence.care_site_id,
        care_site.care_site_name,
        -- visit_occurrence.filename
    FROM omop.person AS person
    LEFT JOIN omop.visit_occurrence AS visit_occurrence
        ON person.person_id = visit_occurrence.person_id
    LEFT JOIN omop.care_site AS care_site
        ON visit_occurrence.care_site_id = care_site.care_site_id
    """

    sql_admission = f"""
    SELECT 
        person.person_id,
        person.gender_concept_id,
        person.birth_datetime,
        visit_occurrence.visit_occurrence_id,
        visit_occurrence.visit_start_datetime,
        visit_occurrence.visit_end_datetime,
        visit_occurrence.admitted_from_source_value,
        visit_occurrence.care_site_id,
        -- visit_occurrence.filename
    FROM omop.person AS person
    LEFT JOIN omop.visit_occurrence AS visit_occurrence
        ON person.person_id = visit_occurrence.person_id
    """

    # check if age is being set as true for extraction
    if features['static']['age']['extract'] is True:
        # get age
        admissions = pd.read_sql(sql=sql_admission_w_care_site, con=conn)
        admissions['age'] = (admissions['visit_start_datetime'] - admissions['birth_datetime']).dt.days / 365.25
        admissions['age'] = pd.to_numeric(admissions['age'], errors='coerce')
        admissions['age'] = admissions['age'].round().astype('Int64')
    else:
        return "Age is not defined as feature, stop extraction"

    # check if gender is set as true
    if features['static']['gender']['extract'] is True:
        # preprocess gender
        admissions['gender_concept_id'] = admissions['gender_concept_id'].replace({
            'Man': 'male',
            'Vrouw': 'female',
            'M': 'male',
            'V': 'female'
        })
    else:
        # remove gender column from dataframe if configured in yaml file
        admissions = admissions.drop(columns=['gender_concept_id'])

    # convert filename to hospital_ehr
    # Define the hospital and EHR names to look for
    hospitals = ['amc', 'vumc', 'aumc', 'rumc', 'mcl', 'olvg', 'umcu', 'cze', 'radboud']
    ehrs = ['epic', 'metavision', 'hix']

    def extract_db_origin(filename):
        """Extract hospital and EHR from filename to create db_origin"""
        if pd.isna(filename):
            return None
        
        # Convert to lowercase for case-insensitive matching
        filename_lower = str(filename).lower()
        
        # Find hospital name
        hospital = None
        for h in hospitals:
            if h in filename_lower:
                hospital = h
                break
        
        # Find EHR name
        ehr = None
        for e in ehrs:
            if e in filename_lower:
                ehr = e
                break
        
        # Combine if both found
        if hospital and ehr:
            return f"{hospital}_{ehr}"
        elif hospital:
            return hospital
        elif ehr:
            return ehr
        else:
            return None

    # Apply the function to create the new column
    admissions['db_origin'] = admissions['filename'].apply(extract_db_origin)

    # Show the results
    print(admissions[['filename', 'db_origin']].head(20))
    print(f"\nUnique db_origin values:\n{admissions['db_origin'].value_counts()}")

    admissions.to_csv(ROOT + '/data/raw/admissions.csv')
    print("")
    print("admissions csv has been saved")
    print("")
    return admissions


def get_weight(conn, features):
    
    # extract if extraction of weight is set as true
    if features['static']['weight']['extract'] is True:

        weight = [3026600, #  Body weight Estimated
                3025315, #  Body weight
                3013762 #  Body weight Measured
                ]
        
        # weight = config['features']['weight']['ids'] # TODO check if these works instead of hardcoded weight ids
        
        static_variable_lists = {
                                'weight': weight 
                                }
        
        # Flatten the list of concept IDs
        weight_concept_ids = ', '.join(map(str, sum(static_variable_lists.values(), [])))

        weight_query = f"""
            SELECT
                m.person_id,
                m.measurement_datetime,
                v.visit_start_datetime,
                (m.measurement_datetime - v.visit_start_datetime) AS timediff,
                m.value_as_number
            FROM omop.measurement AS m
            FULL OUTER JOIN
                omop.visit_occurrence AS v
                ON m.person_id = v.person_id
            WHERE 
                (measurement_concept_id IN ({weight_concept_ids})) AND
                ((m.measurement_datetime - v.visit_start_datetime) <= INTERVAL '3 days')
        """

        weight = pd.read_sql(sql=weight_query, con=conn)
        weight.to_csv(ROOT + '/data/raw/weight.csv')
        print("")
        print("weight csv has been saved")
        print("")

        return weight
    
    else:
        return


def get_height(conn, features):
    
    # extract if extraction of height is set as true
    if features['static']['height']['extract'] is True:

        height = [3035463, #  Body height Estimated
                    3023540, # Body height Measured
                    3036277  #  Body height
                    ]
        
        # height = config['features']['weight']['ids'] # TODO check if these works instead of hardcoded height ids
        
        static_variable_lists = {
                                'height': height 
                                }
        
        # Flatten the list of concept IDs
        height_concept_ids = ', '.join(map(str, sum(static_variable_lists.values(), [])))

        height_query = f"""
            SELECT
                m.person_id,
                m.measurement_datetime,
                v.visit_start_datetime,
                (m.measurement_datetime - v.visit_start_datetime) AS timediff,
                m.value_as_number
            FROM omop.measurement AS m
            FULL OUTER JOIN
                omop.visit_occurrence AS v
                ON m.person_id = v.person_id
            WHERE 
                (measurement_concept_id IN ({height_concept_ids})) AND
                ((m.measurement_datetime - v.visit_start_datetime) <= INTERVAL '3 days')
        """

        height = pd.read_sql(sql=height_query, con=conn)
        height.to_csv(ROOT + '/data/raw/height.csv')
        print("")
        print("height csv has been saved")
        print("")
        return height
    
    else:
        return


def get_ventilated(conn, features, extraction_window):
    
    # extract if ventilation extraction is set as true
    if features['dynamic']['ventilated']['extract'] is True:
        
        # extraction window in hours
        extraction_window = extraction_window

        # declare ventilation ids
        vent_ids = features['dynamic']['ventilated']['ids']

        # Convert the list of concept IDs as one single string
        vent_ids = ', '.join(map(str, vent_ids))

        vent_query = f"""
        SELECT
            m.person_id,
            m.measurement_datetime,
            TRUE as ventilated,
            m.measurement_concept_id,
            m.value_as_number,
            -- m.value_as_concept_id,
            v.visit_start_datetime,
            (m.measurement_datetime - v.visit_start_datetime) AS timediff
        FROM omop.measurement AS m
            FULL OUTER JOIN
                omop.visit_occurrence AS v
                ON m.person_id = v.person_id
        WHERE
            (measurement_concept_id IN ({vent_ids})) AND
            ((m.measurement_datetime - v.visit_start_datetime) <= INTERVAL '{extraction_window} hours') AND
            ((m.measurement_datetime - v.visit_start_datetime) >= INTERVAL '0 hours')
        """

        ventilated = pd.read_sql(sql=vent_query, con=conn)
        ventilated.to_csv(ROOT + '/data/raw/ventilated.csv')
        print("")
        print("ventilated csv has been saved")
        print("")
        return ventilated
    
    else:
        return


def get_measurements(conn, features, extraction_window):
    
    # extraction window in hours
    extraction_window = extraction_window

    # extract features and ids that have their settings set as True for extraction
    list_of_features = []
    list_of_ids = []

    for feature, settings in features['dynamic'].items():
        if settings['extract'] is True:
            list_of_features.append(feature)
            list_of_ids.append(settings['ids'])

    # create new dictionary based on extracted variables
    features_dict = dict(zip(list_of_features, list_of_ids))

    # drop features that have no id to make code below work
    features_dict = {k: v for k, v in features_dict.items() if v is not None}
    
    # Create a single CASE statement for all variables
    case_clauses = []
    for variable_name, concept_ids in features_dict.items():
        case_clauses.append(
            f"WHEN m.measurement_concept_id IN ({', '.join(map(str, concept_ids))}) THEN '{variable_name}'"
        )

    # Combine all CASE clauses into a single CASE statement for use in query
    case_statement = f"CASE {' '.join(case_clauses)} END AS variable_name"

    # Flatten the list of concept IDs as one single string
    all_ids = ', '.join(map(str, sum(features_dict.values(), [])))

    # SQL query, measured within extraction window of admission
    measurements_query = f"""
    SELECT
        m.person_id,
        m.measurement_datetime,
        m.measurement_concept_id,
        m.value_as_number,
        -- m.value_as_concept_id, -> not in this omop version
        {case_statement},
        v.visit_start_datetime,
        (m.measurement_datetime - v.visit_start_datetime) AS timediff
    FROM omop.measurement AS m
        FULL OUTER JOIN
            omop.visit_occurrence AS v
            ON m.person_id = v.person_id
    WHERE 
        (measurement_concept_id IN ({all_ids})) AND
        ((m.measurement_datetime - v.visit_start_datetime) <= INTERVAL '{extraction_window} hours') AND
        ((m.measurement_datetime - v.visit_start_datetime) >= INTERVAL '0 hours')
    """

    print("extracting measurements")
    measurements = pd.read_sql(sql=measurements_query, con=conn)
    measurements.to_csv(ROOT + '/data/raw/measurements.csv')
    print("")
    print("measurements csv has been saved")
    print("")
    return measurements


### --- database connection and close functions below --- ###

def database_connect(path_to_duck_db):

    # connect to database
    conn = duckdb.connect(path_to_duck_db, read_only=True)

    return conn


def close_connection(conn):

    conn.close()

    return

def get_nice_aumc():

    # from NICE itself
    # amc_mv = pd.read_csv(ROOT + "data/raw/nice/amc_mv_nice_data.csv", index_col=False)
    # vumc_mv = pd.read_csv(ROOT + "data/raw/nice/vumc_mv_nice_data.csv", index_col=False) -> use Epic only due to accurate APACHE IV scores
    amc_epic = pd.read_csv(ROOT + "data/raw/nice/nice_icu_data_amc_hashed_clean.csv", index_col=False)
    aumc_epic = pd.read_csv(ROOT + "data/raw/nice/nice_icu_data_aumc_hashed_clean.csv", index_col=False)

    # select necessary columns only TODO fix conversion of icudata hash to person_id and visit_occurrence_id -> no link with visit_occurrence_id, use date of admission?
    # mv_ids = ['PatientID_hashed', 'Hospitalnumber_hashed']
    # epic_ids = ['PSEUDO_PAT_ID', 'PSEUDO_PAT_ENC_CSN_ID']
    mv_ids = ['PatientID_hashed', 'Hospitalnumber_hashed', 'ICOpnameDatumTijd']
    epic_ids = ['PSEUDO_PAT_ID', 'PSEUDO_PAT_ENC_CSN_ID', 'ICOpnameDatumTijd']

    relevant_columns = ['ac_ren_fail', 'dysrhytmia', 'cva', 'gastro_bleed', 'intracran_mass', 'chr_renal_insuf', 'chron_dialysis', 'neoplasm', 'aids', 'hem_malign', 'cirrhosis', 'cardio_vasc_insuf', 'resp_insuf', 'imm_insuf', 'apache_score', 'apache_prob', 'nice_saps_score', 'nice_saps_prob', 'nice_apache_score', 'nice_apache3_score', 'thrombotherap', 'ap4diag1', 'ap4diag2', 'numgraft', 'nice_ap4_aps' , 'nice_ap4_score', 'nice_ap4_prob', 'nice_ap4_model', 'nice_ap4_excluded']

    # amc_mv = amc_mv[mv_ids + relevant_columns].copy()
    # vumc_mv = vumc_mv[relevant_columns].copy()
    amc_epic = amc_epic[epic_ids + relevant_columns].copy() # -> use Epic only due to accurate NICE scores
    aumc_epic = aumc_epic[epic_ids + relevant_columns].copy()

    # # rename identifiers to OMOP schema: person_id and visit_occurrence_id
    # amc_mv = amc_mv.rename(columns={
    #     'PatientID_hashed': 'person_id',
    #     'Hospitalnumber_hashed': 'visit_occurrence_id'
    # })

    # PM convert ICOpnameDatumTijd to visit_start_date and merge on that
    
    amc_epic = amc_epic.rename(columns={
        'PSEUDO_PAT_ID': 'person_id',
        'PSEUDO_PAT_ENC_CSN_ID': 'visit_occurrence_id',
        'ICOpnameDatumTijd': 'visit_start_datetime'
    })
    
    aumc_epic = aumc_epic.rename(columns={
        'PSEUDO_PAT_ID': 'person_id',
        'PSEUDO_PAT_ENC_CSN_ID': 'visit_occurrence_id',
        'ICOpnameDatumTijd': 'visit_start_datetime'
    })

    # concat all files to one aumc dataframe
    # aumc_df = pd.concat([amc_mv, amc_epic, aumc_epic], ignore_index=True)
    aumc_df = pd.concat([amc_epic, aumc_epic], ignore_index=True)

    # drop rows where person_id or visit_occurrence_id is NaN
    aumc_df = aumc_df.dropna(subset=['person_id', 'visit_occurrence_id', 'visit_start_datetime', 'nice_ap4_aps' , 'nice_ap4_score', "nice_ap4_prob", 'nice_ap4_model'])
    
    # count NaNs per row
    aumc_df['nan_count'] = aumc_df.isna().sum(axis=1)
    
    # keep the row with the least NaNs for each combination
    aumc_df = aumc_df.sort_values('nan_count').drop_duplicates(
        subset=['person_id', 'visit_occurrence_id', 'visit_start_datetime'], 
        keep='first'
    )
    
    # drop the helper column
    aumc_df = aumc_df.drop(columns=['nan_count'])

    # get only columns required for modeling
    aumc_df = aumc_df[["person_id", "visit_occurrence_id", 'visit_start_datetime', "nice_ap4_score", "nice_ap4_prob", "nice_ap4_model"]].copy()

    # # hash person_id and visit_occurrence_id for merging with icudata using PySpark
    # spark = SparkSession.builder.appName("hash_ids").getOrCreate()
    
    # # Convert pandas DataFrame to Spark DataFrame
    # spark_df = spark.createDataFrame(aumc_df)
    
    # # Apply xxhash64 to person_id and visit_occurrence_id
    # spark_df = spark_df.withColumn("person_id", xxhash64(col("person_id").cast("string")))
    # spark_df = spark_df.withColumn("visit_occurrence_id", xxhash64(col("visit_occurrence_id").cast("string")))
    
    # # Convert back to pandas
    # aumc_df = spark_df.toPandas()
    
    # # Stop Spark session
    # spark.stop()

    def xxhash64(val):
        unsigned = xxhash.xxh64(str(val), seed=42).intdigest()
    
        # Convert to signed 64-bit to match PySpark output
        return unsigned - 2**64 if unsigned >= 2**63 else unsigned

    aumc_df["person_id"] = aumc_df["person_id"].apply(xxhash64)
    aumc_df["visit_occurrence_id"] = aumc_df["visit_occurrence_id"].apply(xxhash64)

    aumc_df.to_csv(ROOT + "data/intermediate/nice/aumc.csv")

    return


if __name__=="__main__":
    main()