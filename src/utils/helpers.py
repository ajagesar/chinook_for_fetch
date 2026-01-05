import yaml
import pandas as pd

# ROOT = "C:/Users/Ameet/Documents/research/repositories/chinook_for_fetch/" # local
ROOT = "/home/ameet.jagesar@mydre.org/research/chinook_for_fetch/" # mydre

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
    
    df_imputed = pd.DataFrame(imputer.transform(df))
    df_imputed.columns = df.columns
    df_imputed.index = df.index

    return df_imputed