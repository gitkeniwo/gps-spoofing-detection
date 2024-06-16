import pandas as pd
import numpy as np
import math
import random
from haversine import haversine
from estimate_path_err import estimate_path_err
import sys
import os
import copy
from detected import*
# data_unique = data.drop_duplicates(subset=["GPS_lat", "GPS_long","CID"])
# data_unique = data_unique.reset_index(drop=True)
# data_unique["spoofed"] = 0
# data_unique.to_csv("../data/drive-me-not/processed/trace4_unique.csv")
cell_data=pd.read_csv("data/drive-me-not/CellsDatabase.csv")
cell_data=cell_data.rename(columns={'cell':'CID','area':'LAC','net':'MNC'})
def generate_unique(file_path):

    data= pd.read_csv('data/drive-me-not/{}'.format(file_path))
    data_unique = data.drop_duplicates(subset=["GPS_lat", "GPS_long","CID"])
    data_unique = data_unique.reset_index(drop=True)
    data_unique["spoofed"] = 0
    data_unique.to_csv("data/drive-me-not/unique/unique_{}".format(file_path))



def get_bursts(data_unique,data_spoofed,th,time):
    merged=pd.merge(data_unique,data_spoofed,on=["Time"],how='left')
    merged.to_csv('merged.csv')
    # origin_data[["GPS_lat", "GPS_long"]]=merged[["GPS_lat", "GPS_long"]].combine_first(origin_data[["GPS_lat", "GPS_long"]])
    origin_data=copy.deepcopy(data_unique)
    origin_data[["GPS_lat", "GPS_long"]]=merged[["GPS_lat_y", "GPS_long_y"]]
    origin_data['spoofed']=merged['spoofed_y']
    origin_data=origin_data.dropna()
    origin_data = origin_data.reset_index(drop=True)
    mdata=pd.merge(origin_data,cell_data,on=['CID','LAC','MNC'],how='left')
    origin_data['lat']=mdata['lat']
    origin_data['lon']=mdata['lon']
    origin_data=origin_data.dropna()
    origin_data['e_lat']=0.0
    origin_data['e_lon']=0.0
    origin_data['difference']=0.0
    origin_data = estimate_path_err(origin_data, expF=20, qthr=0.9)
    data_spoofed_pos=origin_data.drop_duplicates(subset=["GPS_lat", "GPS_long"])
    data_spoofed_pos=data_spoofed_pos.reset_index(drop=True)
    # diff=origin_data['difference'].values
    # detected_time=origin_data.iloc[np.argwhere(diff>th)[0]]['Time']
    # start_time=origin_data[origin_data['spoofed']==1].iloc[0]['Time']
    item=get_diff(data_spoofed_pos,time,th)
    print(item)


    mdata=pd.merge(data_unique,cell_data,on=['CID','LAC','MNC'],how='left')
    data_unique['lat']=mdata['lat']
    data_unique['lon']=mdata['lon']
    data_unique=data_unique.dropna()
    data_unique['e_lat']=0.0
    data_unique['e_lon']=0.0
    data_unique['difference']=0.0
    data_unique = estimate_path_err(data_unique, expF=20, qthr=0.9)
    data_unique_pos=data_unique.drop_duplicates(subset=["GPS_lat", "GPS_long"])
    data_unique_pos=data_unique_pos.reset_index(drop=True)
    # threshold=np.quantile(data_unique_pos['difference'].values,0.50)
    benigh_bursts=maxima_burst(data_unique_pos,th)
    # print(benigh_bursts)
    fp=0
    times=[]
    singles=0
    for key,value in benigh_bursts.items():
        if value==1:
            singles+=1
        time_diff=data_unique_pos.iloc[key+value-1]['Time']-data_unique_pos.iloc[key]['Time']
        times.append(time_diff)
        # print(time_diff)
        if time_diff>time:
            fp+=1
    print(fp/(len(benigh_bursts)-singles))




    
    # print(benigh_bursts)
    # print(max(times))
    # print(data_unique_pos.shape)
    # data_spoofed_pos=origin_data.drop_duplicates(subset=["GPS_lat", "GPS_long"])
    # data_spoofed_pos=data_spoofed_pos.reset_index(drop=True)
    # bursts=maxima_burst(data_spoofed_pos,threshold)
    # print(bursts)
    return fp/len(benigh_bursts)
def main():

    # file_list=['trace{}.csv'.format(i) for i in range(1,9)]
    # for i in file_list:
    #     print(i)
    #     generate_unique(i)
    file_list=['trace{}.csv'.format(i) for i in range(1,9)]
    # file_list=['trace4.csv']
    for i in file_list:
        data_unique= pd.read_csv('data/drive-me-not/unique/unique_{}'.format(i))
        data_spoof= pd.read_csv('data/drive-me-not/spoofed/spoofed_{}'.format(i))
        get_bursts(data_unique,data_spoof,0.53,78442)
    
if __name__ == "__main__":
    main()