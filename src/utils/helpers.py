import yaml
import pandas as pd

# ROOT = "C:/Users/Ameet/Documents/research/repositories/chinook_for_fetch/" # local
ROOT = "/home/ameet.jagesar@mydre.org/repositories/chinook_for_fetch/" # mydre

with open(ROOT + 'config.yaml', 'r') as file:
    config = yaml.safe_load(file)


with open(ROOT + 'src/apache_iv_calc.yml', 'r') as file_2:
    apache_config = yaml.safe_load(file_2)




def dbs_read_sql(sql:str, list_of_con:list, keep_source_db_id=False):
    list_of_output_per_db = []
    
    for i, conn in enumerate(list_of_con):
        output_per_db = pd.read_sql(sql=sql, con=conn)
        output_per_db['source_db'] = i
        list_of_output_per_db.append(output_per_db)

    query_result = pd.concat(list_of_output_per_db)

    if keep_source_db_id == False:
        query_result = query_result.drop(columns='source_db')

    return query_result


def impute_dataframe(imputer, df):


    # Identify columns that are completely NaN (no valid data at all)
    # These columns will be dropped by sklearn's imputer during transform
    all_nan_cols = df.columns[df.isna().all()]
    
    # Transform the dataframe using the fitted imputer
    # Note: This will only return columns that weren't all-NaN during fitting
    df_imputed = pd.DataFrame(imputer.transform(df))
    
    # Assign column names, but only for the columns that were kept by the imputer
    # We filter out the all-NaN columns to match the actual number of columns returned
    df_imputed.columns = [col for col in df.columns if col not in all_nan_cols]
    
    # Preserve the original row index
    df_imputed.index = df.index
    
    # Manually add back the all-NaN columns that were dropped by the imputer
    # Fill them with a default value (0 in this case, but could be np.nan or another value)
    # Use len() to check if there are any all-NaN columns to avoid ambiguous truth value error
    if len(all_nan_cols) > 0:

        print(f"Warning: Columns with all NaN values: {all_nan_cols}. Will be imputed with 0.")

        for col in all_nan_cols:
            df_imputed[col] = 0  # Choose an appropriate fill value for your use case
    
    # Reorder columns to match the original dataframe's column order
    # This ensures consistency between the input and output dataframes
    df_imputed = df_imputed[df.columns]
    
    return df_imputed