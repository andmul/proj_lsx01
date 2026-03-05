import polars as pl
import glob,os
from pathlib import Path
from datetime import datetime
import pytz
from datetime import time
from datetime import date
import pandas as pd
import json
#import pyarrow
import numpy as np
#from ftplib import FTP


##MAIN

#Step 1: all files in last directory
files = glob.glob(r'c:\temp\lsx\last\lsxtradesyesterday_*.csv')
# Step 2: Read all CSVs into Polars DataFrames

df_list = [pl.read_csv(f, infer_schema_length=100,separator=";",decimal_comma=True) for f in files]
#delete name column from all frames
for index,dlist in enumerate(df_list):
        if len(df_list[index].columns) == 11:
            df_list[index]=df_list[index].drop('displayName')

# Step 3: Concatenate all data into one DataFrame
print('start concatenation....')
df = pl.concat(df_list, how="vertical")
print('...end concatenation')
# Step 4: Column conversions

# -- Convert 'tradeTime' and 'publishedTime' to pl.Datetime
# Parse the tradeTime column to datetime and convert to Berlin time
df = df.with_columns(
     pl.col("tradeTime")
     .str.replace("Z", "")
     .str.replace("T"," ")
     .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("Europe/Berlin")
    .alias("tradeTime"),

    pl.col("publishedTime")
     .str.replace("Z", "")
     .str.replace("T"," ")
     .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("Europe/Berlin")
    .alias("publishedTime")
    
)
# -- Cast 'price' to float and 'size' to int
df = df.with_columns([
    pl.col("price").cast(pl.Float64),
    pl.col("size").cast(pl.Int64),
])

# -- Clean string columns
str_columns = ['isin', 'displayName', 'quotation', 'currency', 'TVTIC', 'mic', 'flags']
for col in str_columns:
    if col in df.columns:
        df = df.with_columns(pl.col(col).cast(pl.Utf8).fill_null("").str.strip_chars())
 
print('Done with structure cleanup')


#df=df.filter(pl.col('tradeTime')>date(2025,3,31))
df_sorted=df.sort('publishedTime')
print('all sorted,now make it unique...')
df=df_sorted.unique(subset=['TVTIC'],keep='last')
print('unicated, now dropping canceled transactions')
df=df.filter(~pl.col('flags').str.contains('CANC'))

df.write_parquet('c:\\temp\\lsx\\raw24_07_23to25_03_30.parquet')
       