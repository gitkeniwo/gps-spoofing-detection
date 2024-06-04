import pandas as pd
import numpy as np
import math
import random

origin_data = pd.read_csv(
    "/Users/liguangyu/Downloads/gps-spoofing-detection-cellular-master/trace4_unique.csv"
)
spoofed_data = pd.read_csv(
    "/Users/liguangyu/Downloads/gps-spoofing-detection-cellular-master/trace4_spoofed.csv"
)
merged=pd.merge(origin_data,spoofed_data,on=["Time"],how='left')
merged.to_csv('merged.csv')
# origin_data[["GPS_lat", "GPS_long"]]=merged[["GPS_lat", "GPS_long"]].combine_first(origin_data[["GPS_lat", "GPS_long"]])
origin_data[["GPS_lat", "GPS_long"]]=merged[["GPS_lat_y", "GPS_long_y"]]
origin_data['spoofed']=merged['spoofed']
origin_data=origin_data.dropna()
origin_data = origin_data.reset_index(drop=True)
origin_data.to_csv('full_spoofed_trace4.csv')