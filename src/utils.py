import numpy as np
import pandas as pd
import torch 
import os 
import tqdm 
import sqlite3


def load_database(database, indices=None) -> pd.DataFrame:
    """Opens the sqlite database and converts it to pandas dataframe. 
    If indices are provided, data is splitted accordingly.

    Returns:
        pd.DataFrame: Dataframe [x, y, label, filename, slide].
    """
    # open database file 
    DB = sqlite3.connect(database)
    
    # command to select all data from the table
    command = 'SELECT coordinateX as x, coordinateY as y, Annotations.agreedClass as label, Slides.filename, Annotations.slide\
                FROM Annotations_coordinates \
                INNER JOIN Annotations \
                ON Annotations_coordinates.annoId = Annotations.uid \
                INNER JOIN Slides \
                ON Annotations.slide = Slides.uid \
                WHERE Annotations.deleted=0'

    # execute the command and create dataframe
    df = pd.read_sql_query(command, DB)

    # load data for current split
    if indices:
        df = df.query('slide in @indices')

    return df 

# train 


# eval


# earlystopping


# metrics 


# transforms