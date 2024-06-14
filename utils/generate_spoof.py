import pandas as pd
import numpy as np
import math
import random
from haversine import haversine

import sys
import os
random.seed(42)
np.random.seed(42)
cell_data=pd.read_csv("data/drive-me-not/CellsDatabase.csv")
def average_speed(data, start):
    time = data.iloc[start]["Time"] - data.iloc[0]["Time"]
    total_length = 0
    for i in range(1, start):
        lats = [data.iloc[i - 1]["GPS_lat"], data.iloc[i]["GPS_lat"]]
        lons = [data.iloc[i - 1]["GPS_long"], data.iloc[i]["GPS_long"]]
        total_length += haversine(lats, lons)
    return total_length / time


def generate_spoof_trace(data, speed, angle, start):
    for i in range(start, data.shape[0] - 1):
        time_interval = data.iloc[i + 1]["Time"] - data.iloc[i]["Time"]
        
        current_speed = speed * (0.75 + np.random.normal(0, 0.5))
        distance = time_interval * current_speed
        delta_lat = distance / 6371 * np.cos(angle)
        delta_lon = (
            distance
            / 6371
            * np.sin(angle)
            / np.cos(np.radians(data.iloc[i]["GPS_lat"]))
        )

        # Calculate new position
        new_lat = data.iloc[i]["GPS_lat"] + np.degrees(delta_lat)
        new_lon = data.iloc[i]["GPS_long"] + np.degrees(delta_lon)
        data.at[i + 1, "GPS_lat"] = new_lat
        data.at[i + 1, "GPS_long"] = new_lon
        data.at[i + 1, "spoofed"] = 1
    return data
def generate_trace(file_path):

    data= pd.read_csv('data/drive-me-not/{}'.format(file_path))
    
    cell_data_lalon=cell_data
    cell_data_lalon=cell_data_lalon.rename(columns={'cell':'CID','area':'LAC','net':'MNC'})
    data_distance=pd.merge(data,cell_data_lalon,on=['CID','LAC','MNC'],how='left')
    data_distance['distance']=data_distance.apply(lambda row: haversine([row['GPS_lat'],row['lat']],[row['GPS_long'],row['lon']]),axis=1)
    data_unique = data.drop_duplicates(subset=["GPS_lat", "GPS_long","CID"])
    data_unique = data_unique.reset_index(drop=True)
    data_unique["spoofed"] = 0
    data_spoofed = data_unique.drop_duplicates(subset=["GPS_lat", "GPS_long"])
    data_spoofed = data_spoofed.reset_index(drop=True)
    start = int(0.5 * data_spoofed.shape[0])
    # print(start)
    total = data_spoofed.iloc[-1]["Time"] - data_spoofed.iloc[0]["Time"]
    remain_time = data_spoofed.iloc[-1]["Time"] - data_spoofed.iloc[start]["Time"]
    speed = average_speed(data_spoofed, start)
    # random pick a direction
    angle = np.random.uniform(0, 2 * np.pi)
    data_spoofed = generate_spoof_trace(data_spoofed, speed, angle, start)
    data_spoofed.to_csv("data/drive-me-not/spoofed/spoofed_{}".format(file_path))

def main():
    file_list=['trace{}.csv'.format(i) for i in range(1,9)]
    for i in file_list:
        print(i)
        generate_trace(i)
    
if __name__ == "__main__":
    main()
