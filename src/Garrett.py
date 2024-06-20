import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from haversine import haversine

th_1=   []
bl_1=   []
th_2=   []
w_2=    []

global_r1=0.7
global_c1=3
global_r2=0
global_c2=18
FP_rate_1 =         []
detection_time_1 =  []
FP_rate_2 =         []
detection_time_2 =  []

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

def Garrett(filename):
    '''Do Garrett's two methods on the file and output the figures.'''
    
    path=r'./data/drive-me-not/processed/'
    output_path=r'./outputs/png/'+filename
    if not os.path.exists(output_path):
        os.mkdir(output_path) 
    full_filename='spoofed_'+filename+'_unique_cell.csv'

    print("Start "+filename)

    data=pd.read_csv(path+full_filename) 
    data=data[['Time','GPS_lat','GPS_long','lat','lon','spoofed']]
    plt.clf()
    plt.scatter(data['GPS_long'], data['GPS_lat'],s=5,label='X_GPS (spoofed)')
    plt.scatter(data['lon'], data['lat'],s=1,label='Y_est (real)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Moving Path of T'+filename[1:])
    plt.legend()
    plt.savefig(output_path+'/Garrett_moving_path')

    #Garrett's first method
    data['d_e']=data.apply(lambda row: haversine((row['GPS_lat'],row['GPS_long']),(row['lat'],row['lon'])), axis=1)
    #Calculate the impact of different thresholds.
    threshold=np.arange(0.5,8,0.2,dtype=float)
    for th in threshold:
        name='th_'+str(th)
        data[name]=data.apply(lambda row: row['d_e']>th, axis=1)



    #Calculate the impact of different bursts length.
    burst_length=np.arange(1,15,1,dtype=int)
    attack_start_id=np.where(data['spoofed']==1)[0][0]
    attack_start_time=data['Time'][attack_start_id]

    #An empty 2d data frame to store the result of FP. rows: burst_length, columns: threshold 
    fp_data=pd.DataFrame(columns=threshold)
    #An empty 2d data frame to store the result of time to detect.
    detect_time_data=pd.DataFrame(columns=threshold)

    for th in threshold:
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
            detect_time=detect_time/1000 #change ms to s
            detect_time_arr=np.append(detect_time_arr,detect_time)

        fp_data[th]=fp_arr
        detect_time_data[th]=detect_time_arr

    #Plot the FP.
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    X, Y = np.meshgrid(threshold, burst_length)
    Z = fp_data

    ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
    ax3.view_init(30, 110)

    ax3.set_xlabel('threshold')
    ax3.set_ylabel('Burst Length')
    ax3.set_zlabel('')
    ax3.legend([],title=filename)
    plt.title('Threshold Method\n False Positive Rate')
    plt.savefig(output_path+'/Garrett_FP_rate')
    plt.close(fig)

    
    #Plot the detection time.
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    X, Y = np.meshgrid(threshold, burst_length)
    Z = detect_time_data

    ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
    ax3.view_init(30,110)

    ax3.set_xlabel('threshold')
    ax3.set_ylabel('Burst Length')
    ax3.set_zlabel('')
    ax3.legend([],title=filename)
    plt.title('Threshold Method\n Detection Time (s)')
    plt.savefig(output_path+'/Garrett_detection_time')
    plt.close(fig)


    detect_time_norm=(detect_time_data-detect_time_data.stack().min())/(detect_time_data.stack().max()-detect_time_data.stack().min())
    arith_mean=detect_time_norm+fp_data/2

    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    X, Y = np.meshgrid(threshold, burst_length)
    Z = arith_mean

    ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
    ax3.view_init(30,110)

    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Burst Length')
    ax3.set_zlabel('')
    ax3.legend([],title=filename)
    plt.title('Threshold Method\n Arithmetic Mean')
    plt.savefig(output_path+'/Garrett_detection_arithmetic_mean')
    plt.close(fig)

    c,r=arith_mean.stack().idxmin()

    
    with open(output_path+"/result.txt","w") as f:
        f.write("Recommended parameter values:\n\n")
        f.write("threshold of Threshold method: "+str(r)+'\n')
        f.write("burst length of Threshold method: "+str(c)+'\n')
        f.write("FP rate of Threshold method: "+str(fp_data[r][c])+'\n')
        f.write("detection time of Threshold method: "+str(detect_time_data[r][c]+c/10)+'\n')
    th_1.append(r)
    bl_1.append(c)
    FP_rate_1.append(fp_data[global_r1][global_c1])
    detection_time_1.append(detect_time_data[global_r1][global_c1]+c/10)
    


    #Garrett's CUSUM method
    threshold=np.arange(0,12,0.2,dtype=float)
    weight=np.arange(0.5,7,0.2,dtype=float)

    #An empty 2d data frame to store the result of FP. rows: weight, columns: threshold 
    fp_data=pd.DataFrame(columns=threshold)
    #An empty 2d data frame to store the result of time to detect.
    detect_time_data=pd.DataFrame(columns=threshold)

    for th in threshold:
        name='d_e'
        fp_arr=np.array([])
        detect_time_arr=np.array([])

        for w in weight:
            num_fp=0
            detect_time=data['Time'][data.shape[0]-1]-attack_start_time
            
            s=0
            for i in range(data.shape[0]):
                s=max(0,s+data[name][i]-w)
                if s>th:
                    if i<=attack_start_id:
                        num_fp=num_fp+1
                    else:
                        detect_time=data['Time'][i]-attack_start_time
                        break
            fp_rate=num_fp/attack_start_id
            fp_arr=np.append(fp_arr,fp_rate)
            detect_time=detect_time/1000 #change ms to s
            detect_time_arr=np.append(detect_time_arr,detect_time)

        fp_data[th]=fp_arr
        detect_time_data[th]=detect_time_arr

    #Plot the FP.
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    X, Y = np.meshgrid(threshold, weight)
    Z = fp_data

    ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
    ax3.view_init(30, 150)

    ax3.set_xlabel('threshold')
    ax3.set_ylabel('Weight')
    ax3.set_zlabel('')
    ax3.legend([],title=filename)
    plt.title('CUSUM Method\n False Positive Rate')
    plt.savefig(output_path+'/Garrett_FP_rate_CUSUM')
    plt.close(fig)


    #Plot the detection time.
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    X, Y = np.meshgrid(threshold, weight)
    Z = detect_time_data

    ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
    ax3.view_init(30,150)

    ax3.set_xlabel('threshold')
    ax3.set_ylabel('Weight')
    ax3.set_zlabel('')
    ax3.legend([],title=filename)
    plt.title('CUSUM Method\n Detection Time (s)')
    plt.savefig(output_path+'/Garrett_detection_time_CUSUM')
    plt.close(fig)
    
    detect_time_norm=(detect_time_data-detect_time_data.stack().min())/(detect_time_data.stack().max()-detect_time_data.stack().min())
    arith_mean=detect_time_norm+fp_data/2

    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    X, Y = np.meshgrid(threshold, weight)
    Z = arith_mean

    ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap='rainbow')
    ax3.view_init(30,150)

    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Weight')
    ax3.set_zlabel('')
    ax3.legend([],title=filename)
    plt.title('CUSUM Method\n Arithmetic Mean')
    plt.savefig(output_path+'/Garrett_detection_CUSUM_arithmetic_mean')
    plt.close(fig)

    c,r=arith_mean.stack().idxmin()
    
    with open(output_path+"/result.txt","a") as f:
        f.write("Recommended parameter values:\n\n")
        f.write("threshold of CUSUM method: "+str(r)+'\n')
        f.write("weight of CUSUM method: "+str(c)+'\n')
        f.write("FP rate of CUSUM method: "+str(fp_data[r][c])+'\n')
        f.write("detection time of CUSUM method: "+str(detect_time_data[r][c])+'\n')
    th_2.append(r)
    w_2.append(c)
    FP_rate_2.append(fp_data[global_r2][global_c2])
    detection_time_2.append(detect_time_data[global_r2][global_c2])

    print("Finished "+filename)

    

if __name__=='__main__':
    for i in range(1,9):
        filename='trace'+str(i)
        Garrett(filename)

    output_path_total=r'./outputs/png/total/'
    with open(output_path_total+"/result.txt","a") as f:
        f.write("threshold of Threshold method:\n")
        f.write(str(th_1)+"\n")
        f.write("burst length of Threshold method: \n")
        f.write(str(bl_1)+"\n")
        f.write("threshold of CUSUM method:\n")
        f.write(str(th_2)+"\n")
        f.write("weight of CUSUM method:\n")
        f.write(str(w_2)+"\n")
        f.write("FP rate of Threshold method:\n")
        f.write(str(FP_rate_1)+"\n")
        f.write("detection time of Threshold method:\n")
        f.write(str(detection_time_1)+"\n")
        f.write("FP rate of CUSUM method:\n")
        f.write(str(FP_rate_2)+"\n")
        f.write("detection time of CUSUM method:\n")
        f.write(str(detection_time_2)+"\n")