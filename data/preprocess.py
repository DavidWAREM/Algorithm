# data/preprocess.py
import pandas as pd

def preprocess_data(data):
    # Find the index of the row containing 'stop'

    #Definiert den Stopindex, bisdahin  wird nicht gelesen
    stopindex = 'stop'

    stop_index = data[data.apply(lambda row: row.astype(str).str.contains(stopindex).any(), axis=1)].index[0]

    # Drop all rows before the stop index
    processed_data = data.loc[stop_index+1:].reset_index(drop=True)
    return processed_data
