import pandas as pd
import numpy as np
import math

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



data= pd.read_csv("/Users/liguangyu/Downloads/gps-spoofing-detection-cellular-master/full_spoofed_trace4.csv")
cell_data=pd.read_csv("/Users/liguangyu/Downloads/gps-spoofing-detection-cellular-master/Data/gsm_CellsDatabase.csv")
# print(data.head())
# print(data.index)
# lat=data['GPS_lat']
# uniques_lat=np.unique(lat.values)
# long=data['GPS_long']
# uniques_long=np.unique(long.values)
# time=data['Time']
# uniques_time=np.unique(time.values)
# print(len(uniques_long))
# print(len(uniques_lat))
# print(len(uniques_time))
# print(type(lat.values))
# old_lat=data['GPS_lat'][0]
# old_long=data['GPS_long'][0]
# for i in range(1,len(data)):
#     new_lat=data['GPS_lat'][i]
#     new_long=data['GPS_long'][i]
#     if (old_lat!=new_lat and old_long==new_long) or (old_lat==new_lat and old_long!=new_long):
#         print(i)
#     old_lat=new_lat
#     old_long=new_long
# values=data['dBm'].values
# sort_values=np.unique(values)
# print(sorted(sort_values))
# clon=cell_data[cell_data['cell']==24222]

# count=cell_data['radio'].value_counts()
# print(count)
# gsm_cell=cell_data[cell_data['radio']=='GSM']
# gsm_cell.to_csv('gsm_cellsdatabase.csv')


cell_data_lalon=cell_data
cell_data_lalon=cell_data_lalon.rename(columns={'cell':'CID','area':'LAC','net':'MNC'})
# print(data['LAC'][:10])
print(cell_data_lalon['CID'][:10])
print(cell_data_lalon.columns)
mdata=pd.merge(data,cell_data_lalon,on=['CID','LAC','MNC'],how='left')
data['lat']=mdata['lat']
data['lon']=mdata['lon']
data.to_csv('spoofed_trace4_unique_cell.csv')

# data['distance']=data.apply(lambda row: haversine([row['GPS_lat'],row['lat']],[row['GPS_long'],row['lon']]),axis=1)
# data.to_csv('trace4_cell.csv')
# for i in range(1,len(data)):
#     print(i)
#     cid=data['CID'][i]
#     lat=data['GPS_lat'][i]
#     lon=data['GPS_long'][i]
#     clon=cell_data[cell_data['cell']==cid]['lon']
#     clat=cell_data[cell_data['cell']==cid]['lat']
#     # print(type(clon))
#     # break
#     data.loc[i,'Clon']=clon.iloc[0]
#     data.loc[i,'Clon']=clat.iloc[0]
#     data['distance']=haversine([lat,clat.iloc[0]],[lon,clon.iloc[0]])
#     data.to_csv('trace4_cell.csv')
