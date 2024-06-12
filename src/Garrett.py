import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from haversine import haversine

def all_is_true(x):
    for i in x:
        if i==False:
            return False
    return True

def sum_larger_than_thredshold(x,th):
    s=0
    for i in x:
        s=s+i
    if s>th:
        return True
    else:
        return False

path=r'../data/drive-me-not/processed/'
filename=r'spoofed_trace4_unique_cell.csv'
data=pd.read_csv(path+filename) 
data=data[['Time','GPS_lat','GPS_long','lat','lon','spoofed']]

plt.scatter(data['GPS_long'], data['GPS_lat'],s=5,label='X_GPS (spoofed)')
plt.scatter(data['lon'], data['lat'],s=1,label='Y_est (real)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Moving Path')
plt.legend()
plt.show()

#Garrett's first method
data['d_e']=data.apply(lambda row: haversine((row['GPS_lat'],row['GPS_long']),(row['lat'],row['lon'])), axis=1)
#Calculate the impact of different thresholds.
threadshold=np.arange(0.5,11,0.5,dtype=float)
for th in threadshold:
    name='th_'+str(th)
    data[name]=data.apply(lambda row: row['d_e']>th, axis=1)



#Calculate the impact of different bursts length.
burst_length=np.arange(1,20,1,dtype=int)
attack_start_id=np.where(data['spoofed']==1)[0][0]
attack_start_time=data['Time'][attack_start_id]

#An empty 2d data frame to store the result of FP. rows: burst_length, columns: threadshold 
fp_data=pd.DataFrame(columns=threadshold)
#An empty 2d data frame to store the result of time to detect.
detect_time_data=pd.DataFrame(columns=threadshold)

for th in threadshold:
    name='th_'+str(th)
    fp_arr=np.array([])
    detect_time_arr=np.array([])

    for bl in burst_length:
        num_fp=0
        detect_time=data['Time'][data.shape[0]-1]-attack_start_time

        for i in range(data.shape[0]-bl+1):
            if all_is_true(data[name][i:i+bl]):
                if i<=attack_start_id:
                    num_fp=num_fp+1
                else:
                    detect_time=data['Time'][i+bl-1]-attack_start_time
                    break
        fp_rate=num_fp/attack_start_id
        fp_arr=np.append(fp_arr,fp_rate)
        detect_time_arr=np.append(detect_time_arr,detect_time)

    fp_data[th]=fp_arr
    detect_time_data[th]=detect_time_arr


fig = plt.figure()
ax3 = plt.axes(projection='3d')

X, Y = np.meshgrid(threadshold, burst_length)
Z = fp_data

ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
ax3.view_init(30, 110)

ax3.set_xlabel('Threadshold')
ax3.set_ylabel('Burst Length')
ax3.set_zlabel('')
plt.title('False Positive Rate')

plt.show()


fig = plt.figure()
ax3 = plt.axes(projection='3d')

X, Y = np.meshgrid(threadshold, burst_length)
Z = detect_time_data

ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
ax3.view_init(30,110)

ax3.set_xlabel('Threadshold')
ax3.set_ylabel('Burst Length')
ax3.set_zlabel('')
plt.title('Detection Time (s)')

plt.show()

#Garrett's CUSUM method
threadshold=np.arange(5,130,5,dtype=int)
window_length=np.arange(2,100,5,dtype=int)

#An empty 2d data frame to store the result of FP. rows: window_length, columns: threadshold 
fp_data=pd.DataFrame(columns=threadshold)
#An empty 2d data frame to store the result of time to detect.
detect_time_data=pd.DataFrame(columns=threadshold)

for th in threadshold:
    name='d_e'
    fp_arr=np.array([])
    detect_time_arr=np.array([])

    for wl in window_length:
        num_fp=0
        detect_time=data['Time'][data.shape[0]-1]-attack_start_time

        for i in range(data.shape[0]-wl+1):
            if sum_larger_than_thredshold(data[name][i:i+wl],th):
                if i<=attack_start_id:
                    num_fp=num_fp+1
                else:
                    detect_time=data['Time'][i+wl-1]-attack_start_time
                    break
        fp_rate=num_fp/attack_start_id
        fp_arr=np.append(fp_arr,fp_rate)
        detect_time_arr=np.append(detect_time_arr,detect_time)

    fp_data[th]=fp_arr
    detect_time_data[th]=detect_time_arr


fig = plt.figure()
ax3 = plt.axes(projection='3d')

X, Y = np.meshgrid(threadshold, window_length)
Z = fp_data

ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
ax3.view_init(30, 30)

ax3.set_xlabel('Threadshold')
ax3.set_ylabel('Window Length')
ax3.set_zlabel('')
plt.title('CUSUM Method\n False Positive Rate')

plt.show()


fig = plt.figure()
ax3 = plt.axes(projection='3d')

X, Y = np.meshgrid(threadshold, window_length)
Z = detect_time_data

ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
ax3.view_init(30,30)

ax3.set_xlabel('Threadshold')
ax3.set_ylabel('Window Length')
ax3.set_zlabel('')
plt.title('CUSUM Method\n Detection Time (s)')

plt.show()