from utils.helpers import ROOT, config
import duckdb
import pandas as pd

def main():

    conn = database_connect(path_to_duck_db=config['icudata_external'])
    features = config['features']
    extraction_window = config['window_in_hours']

    mortality = get_mortality(conn, features)
    

    close_connection(conn=conn)

    return


def get_mortality(conn, features):

        # check if mortality is being set as true for extraction
    if features['static']['mortality']['extract'] is True:

        sql_death = f"""
            SELECT 
                person_id,
                death_datetime
            FROM death
        """

        sql_mortuary = f"""
            SELECT 
                person_id,
                -- visit_start_datetime,
                visit_end_datetime AS death_datetime
            FROM visit_occurrence
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



### --- database connection and close functions below --- ###

def database_connect(path_to_duck_db):

    # connect to database
    conn = duckdb.connect(path_to_duck_db)

    return conn



def close_connection(conn):

    conn.close()

    return

if __name__=="__main__":
    main()