import numpy as np
import pandas as pd
from scipy.stats import expon
import math
# from haversine import haversine
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
            data.loc[j,'e_lat']=epos[0]
            data.loc[j,'e_lon']=epos[1]
            data.loc[j,'difference']=d
        # nAnchors = Drxtf.shape[0]
        # enXY.append([Drxt.iloc[0]['Time'], *epos, *rpos, d, nAnchors])

    # enXY_df = pd.DataFrame(enXY, columns=['timepoint', 'est_lat', 'est_lon', 'real_lat', 'real_lon', 'distance', 'nAnchors'])
    return data

# Example usage:
# Assuming Drx is a pandas DataFrame with the described attributes
data = pd.read_csv(
    "/Users/liguangyu/Downloads/gps-spoofing-detection-cellular-master/spoofed_trace4_unique_cell.csv"
)
data['e_lat']=0.0
data['e_lon']=0.0
data['difference']=0.0
data = estimate_path_err(data, expF=20, qthr=0.9)
data.to_csv('estimated_spoofed_trace4.csv')
print(np.argmax(data['difference'].values))