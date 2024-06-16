import pandas as pd
import numpy as np
import math
import random
from haversine import haversine
from estimate_path_err import estimate_path_err
import sys
import os
import copy
def maxima_burst(data,th):
    benigh_diff=data['difference'].values
    diff_indexs=np.argwhere(benigh_diff>th)
    # print(diff_indexs)
    diff_indexs=np.squeeze(diff_indexs)
    sorted_index=np.sort(diff_indexs)
    # print(sorted_index)
    burst={}
    prev=sorted_index[0]
    begin=sorted_index[0]
    sequence_length=1
    for i in range(1,sorted_index.shape[0]):
        if sorted_index[i]!=prev+1:
            # burst.append({begin:sequence_length})
            burst[begin]=sequence_length
            sequence_length=1
            prev=sorted_index[i]
            begin=prev
        else:
            sequence_length+=1
            prev=sorted_index[i]
    burst[begin]=sequence_length
    return burst
def get_detected_time(data,bursts):
    max_time=0
    for key,value in bursts.items():
        begin_time=data['Time'][key]
       
        end_time=data['Time'][key+value-1]
        
        time=end_time-begin_time
       
        if max_time<time:
            max_time=time
    return max_time
def get_diff(data,time,threshold):
    anomaly_start_index=data[data['spoofed']==1].index[0]
    anomaly_start_time=data.iloc[anomaly_start_index]['Time']
    diff=data[data['spoofed']==1]['difference'].values
    
    anomaly_detected_index=np.argwhere(diff>threshold)[0][0]
    # print(anomaly_detected_index)
    start_time_point=data.iloc[anomaly_start_index+anomaly_detected_index]['Time']
    duration_detected=0
    diff_detected=0
    for i in range(anomaly_detected_index,data.shape[0]):
        current_time=data.iloc[i]['Time']
        
        if current_time-start_time_point<time:
            continue
        else:
            # print(current_time)
            # print(i)
            duration_detected=current_time-anomaly_start_time
            diff_detected=data.iloc[i]['difference']
            break
    return duration_detected,diff_detected

