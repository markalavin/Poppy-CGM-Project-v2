# This module gets the data -- glucose levels and "records" (meals, insulin, etc.) that is
# used to make a prediction about the next period of glucose levels.

import pandas as pd
import numpy as np
from datetime import timedelta

# This function returns information about Records of events like meals, insulin, etc. in
# the form of a Pandas dataframe with columns time (timestamp), record_type (string) and
# record_amount (number).

def ask_yes_no( question : str ) -> bool:
    while True:
        print( question, "(y/n)" )
        response = input().strip().upper();
        if response == 'Y':
            return True
        elif response == 'N':
            return False

def ask_number( question : str ) -> int:
    while True:
        print( question, "(number)" )
        response = input().strip()
        try:
            return int( response )
        except:
            pass

def record( time : pd.Timestamp, record_type: str, record_amount : int ) -> pd.DataFrame:
    result = pd.DataFrame( [ { 'time' : time, 'record_type' : record_type, 'record_amount' : record_amount } ] )
    return result

def concatenate( df_1 : pd.DataFrame, df_2 : pd.DataFrame ) -> pd.DataFrame:
    return df_2 if df_1.shape[ 0 ] == 0 else pd.concat( [ df_1, df_2 ] )

def get_recent_records( as_of_time : pd.Timestamp ) -> pd.DataFrame:
    print("get_recent_records as_of_time", as_of_time )
    result_df = pd.DataFrame( columns = [ 'time', 'record_type', 'record_amount' ] )

    while ask_yes_no( f"Any more records (meals, insulin, etc.) in four hours leading up to {as_of_time}?" ):

        if ask_yes_no( f"Meal?" ):
            minutes_ago = ask_number( f"How many minutes ago?" )
            meal_time = as_of_time - timedelta( minutes = minutes_ago )
            meal_amount = 9  # Hardwire for now
            result_df = concatenate( result_df, record( meal_time, 'meal', meal_amount ) )

        elif ask_yes_no( f"insulin" ):
            minutes_ago = ask_number( f"How many minutes ago?" )
            insulin_time = as_of_time - timedelta( minutes = minutes_ago )
            insulin_amount = ask_number( f"How many units?" )
            result_df = concatenate( result_df, record( insulin_time, 'insulin', insulin_amount ) )

        elif ask_yes_no(f"minimeal?"):
            minutes_ago = ask_number(f"How many minutes ago?")
            minimeal_time = as_of_time - timedelta(minutes=minutes_ago)
            minimeal_amount = 2  # Hardwire for now
            result_df = concatenate( result_df, record( minimeal_time, 'minimeal', minimeal_amount ) )

        elif ask_yes_no( f"karo?" ):
            minutes_ago = ask_number( f"How many minutes ago?" )
            karo_time = as_of_time - timedelta( minutes = minutes_ago )
            karo_amount = 1  # Hardwire for now
            result_df = concatenate( result_df, record( karo_time, 'karo', karo_amount ) )

        elif ask_yes_no( f"exercise?" ):
            minutes_ago = ask_number(f"How many minutes ago?")
            exercise_time = as_of_time - timedelta(minutes=minutes_ago)
            exercise_amount = 15  # Hardwire for now
            result_df = concatenate( result_df, record( exercise_time, 'exercise', exercise_amount ) )

    return result_df

print( "result of get_recent_records:\n", get_recent_records( pd.Timestamp.now() ) )