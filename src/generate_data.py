import pandas as pd
import numpy as np
import math
import random
from scipy.stats import expon
import sys
import os
import copy

def haversine(lat, lon):
    # Radius of the Earth in kilometers
    R = 6371

    # Convert degrees to radians
    lat1 = math.radians(lat[0])
    lon1 = math.radians(lon[0])
    lat2 = math.radians(lat[1])
    lon2 = math.radians(lon[1])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance

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

def estimate_path_err(Drx, expF, qthr):
    print(f'Estimating path error for {expF}... ')
    enXY = []
    i = 0

    while i < Drx.shape[0]:
        real_lat=Drx.loc[i,'GPS_lat']
        real_long=Drx.loc[i,'GPS_long']
        start=i
        # Find all rows with the same timepoint
        idx = Drx['Time'] == Drx.iloc[i]['Time']
        print(idx)
        # Jump to the row with a new time
        i += idx.sum()

        # Extract RSSI values associated with each anchor
        Drxt = Drx[idx]

        # Find unique rows based on specific columns (area, cell_id, cell_lat, cell_lon)
        Drxt_unique = Drxt.drop_duplicates(subset=[ 'CID', 'lat', 'lon'])

        # Compute the centroid from the anchor coordinates
        # [cell_lat, cell_lon, dBm]
        Drxtf = Drxt_unique[['lat', 'lon', 'dBm']].values

        # Compute weights
        x = np.sort(Drxtf[:, 2])[::-1]  # Sort in descending order
        xn = -(x - x[0])
        y = expon.pdf(xn, scale=expF)
        w = y / y.sum()

        # Real position as mean of the logged values
        # rpos = Drxt[['lat', 'lon']].mean().values

        # Estimated position as the weighted mean
        epos = np.dot(Drxtf[:, :2].T, w)

        # Distance (Km) between estimation and real position
        d = haversine([epos[0],real_lat],[epos[1],real_long])
        for j in range(start,i):
            Drx.loc[j,'e_lat']=epos[0]
            Drx.loc[j,'e_lon']=epos[1]
            Drx.loc[j,'difference']=d
        # nAnchors = Drxtf.shape[0]
        # enXY.append([Drxt.iloc[0]['Time'], *epos, *rpos, d, nAnchors])

    # enXY_df = pd.DataFrame(enXY, columns=['timepoint', 'est_lat', 'est_lon', 'real_lat', 'real_lon', 'distance', 'nAnchors'])
    return Drx

def generate_data(filename):
    '''Grnerate unique spoofed data with cell estimated data by the file name.'''
    path=r'./data/drive-me-not/'
    data= pd.read_csv(path+filename+".csv")
    cell_data=pd.read_csv(path+"CellsDatabase.csv")

    cell_data_lalon=cell_data
    cell_data_lalon=cell_data_lalon.rename(columns={'cell':'CID','area':'LAC','net':'MNC'})
    data_distance=pd.merge(data,cell_data_lalon,on=['CID','LAC','MNC'],how='left')
    data_distance['distance']=data_distance.apply(lambda row: haversine([row['GPS_lat'],row['lat']],[row['GPS_long'],row['lon']]),axis=1)
    # data_distance.to_csv('../data/drive-me-not/processed/trace4_cell.csv')

    data_unique = data.drop_duplicates(subset=["GPS_lat", "GPS_long","CID"])
    data_unique = data_unique.reset_index(drop=True)
    data_unique["spoofed"] = 0
    # data_unique.to_csv("../data/drive-me-not/processed/trace4_unique.csv")

    data_spoofed = data_unique.drop_duplicates(subset=["GPS_lat", "GPS_long"])
    data_spoofed = data_spoofed.reset_index(drop=True)
    start = int(0.5 * data_spoofed.shape[0])
    total = data_spoofed.iloc[-1]["Time"] - data_spoofed.iloc[0]["Time"]
    remain_time = data_spoofed.iloc[-1]["Time"] - data_spoofed.iloc[start]["Time"]
    speed = average_speed(data_spoofed, start)
    # random pick a direction
    angle = np.random.uniform(0, 2 * np.pi)
    data_spoofed = generate_spoof_trace(data_spoofed, speed, angle, start)
    # data_spoofed.to_csv("../data/drive-me-not/processed/trace4_spoofed.csv")

    # origin_data = pd.read_csv(
    #     "/Users/liguangyu/Downloads/gps-spoofing-detection-cellular-master/trace4_unique.csv"
    # )
    # spoofed_data = pd.read_csv(
    #     "/Users/liguangyu/Downloads/gps-spoofing-detection-cellular-master/trace4_spoofed.csv"

    # )

    merged=pd.merge(data_unique,data_spoofed,on=["Time"],how='left')
    merged.to_csv('merged.csv')
    # origin_data[["GPS_lat", "GPS_long"]]=merged[["GPS_lat", "GPS_long"]].combine_first(origin_data[["GPS_lat", "GPS_long"]])
    origin_data=copy.deepcopy(data_unique)
    origin_data[["GPS_lat", "GPS_long"]]=merged[["GPS_lat_y", "GPS_long_y"]]
    origin_data['spoofed']=merged['spoofed_y']
    origin_data=origin_data.dropna()
    origin_data = origin_data.reset_index(drop=True)
    mdata=pd.merge(origin_data,cell_data_lalon,on=['CID','LAC','MNC'],how='left')
    origin_data['lat']=mdata['lat']
    origin_data['lon']=mdata['lon']
    origin_data=origin_data.dropna()
    origin_data.to_csv('./data/drive-me-not/processed/spoofed_'+filename+'_unique_cell.csv')
    # origin_data.to_csv('full_spoofed_trace4.csv')

if __name__=='__main__':
    for i in range(1,9):
        filename='trace'+str(i)
        generate_data(filename)