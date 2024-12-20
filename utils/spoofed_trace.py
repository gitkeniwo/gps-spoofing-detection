import pandas as pd
import numpy as np
import math
import random
from process_speed import haversine

random.seed(42)


def spoofed_start_points(data):

    start = int(random.random() * data.shape[0])
    return start

# average speed km/ms
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


if __name__ == "__main__":

    num = 1
    
    data = pd.read_csv(
        "../data/drive-me-not/trace"+str(num)+".csv"
    )
   
    # data=data[['GPS_lat','GPS_long','CID']]
    data = data.drop_duplicates(subset=["GPS_lat", "GPS_long"])
    data = data.reset_index(drop=True)
    data.to_csv("trace4_unique.csv")
    print(spoofed_start_points(data))
    data["spoofed"] = 0

    # start = spoofed_start_points(data)
    # spoofed start from the middle of trace
    
    start = int(0.5 * data.shape[0])
    print(start)
    total = data.iloc[-1]["Time"] - data.iloc[0]["Time"]
    remain_time = data.iloc[-1]["Time"] - data.iloc[start]["Time"]
    speed = average_speed(data, start)
    
    # random pick a direction
    angle = np.random.uniform(0, 2 * np.pi)
    data = generate_spoof_trace(data, speed, angle, start)
    
    data.to_csv("../data/drive-me-not/spoofed_trace"+str(num)+".csv", index=False)

