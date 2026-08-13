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
        meas_all.groupby('care_site_name')
        .apply(lambda group: group.isnull().mean() * 100)
        .T  # Transpose to put features as rows
        .round(2)
    )

    print(missing_by_group)

    return


def check_raw_tables(conn):

    # Query to find all Albumin concept IDs that actually exist in your measurement table
    albumin_query = """
    SELECT DISTINCT 
        m.measurement_concept_id, 
        c.concept_name,
        c.vocabulary_id,
        c.standard_concept
    FROM omop.measurement AS m
    JOIN omop.concept AS c 
        ON m.measurement_concept_id = c.concept_id
    WHERE 
        c.domain_id = 'Measurement' AND
        c.concept_name ILIKE '%albumin%'
    ORDER BY c.concept_name;
    """

    print("Searching for Albumin concept IDs...")
    albumin_concepts = pd.read_sql(sql=albumin_query, con=conn)

    # Display the findings
    print(albumin_concepts)

    # Extract list of IDs directly to pass into your `all_ids` list
    albumin_ids = albumin_concepts["measurement_concept_id"].tolist()
    print("\nFound Concept IDs:", albumin_ids)

    return


if __name__=="__main__":
    main()