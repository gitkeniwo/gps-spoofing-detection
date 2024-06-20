from matplotlib import pyplot as plt
import numpy as np

output_path="./outputs/png/total/"
traces = []
for i in range(1,9):
    traces.append("trace"+str(i))
FP_rate_1 =         [0.07941701368233195, 0.1083042192958882, 0.09498031496062992, 0.029570747217806042, 0.048131080389144903, 0.28915171288743885, 0.03637059724349158, 0.11085235211923615]
detection_time_1 =  [0.0, 0.2, 0.5, 21.007, 12.744, 0.6, 76.618, 0.3]
FP_rate_2 =         [0.06127305175490779, 0.0, 0.043799212598425195, 0.11955484896661367, 0.0, 0.04119086460032626, 0.0, 0.0]
detection_time_2 =  [0.0, 148.005, 188.494, 105.257, 248.141, 43.854, 106.588, 226.176]

FP_rate_1=np.around(np.array(FP_rate_1)*100,2)
detection_time_1=np.around(np.array(detection_time_1),2)
FP_rate_2=np.around(np.array(FP_rate_2)*100,2)
detection_time_2=np.around(np.array(detection_time_2),2)

x = np.arange(len(traces))
width = 0.2

FP1_x = x
FP2_x = x + width

plt.bar(FP1_x,FP_rate_1,width=width,label="Threshold Method")
plt.bar(FP2_x,FP_rate_2,width=width,label="CUSUM Method")
plt.xticks(x + width,labels=traces)

for i in range(len(traces)):
    plt.text(FP1_x[i],FP_rate_1[i], FP_rate_1[i],va="bottom",ha="center",fontsize=8)
    plt.text(FP2_x[i],FP_rate_2[i], FP_rate_2[i],va="bottom",ha="center",fontsize=8) 

plt.xlabel('Dataset')
plt.ylabel('')
plt.title('False Positive Rate (%)')
plt.legend(loc="upper left")
plt.savefig(output_path+'FP_rate')
plt.show()
plt.close()

DT1_x = x
DT2_x = x + width

plt.bar(DT1_x,detection_time_1,width=width,label="Threshold Method")
plt.bar(DT2_x,detection_time_2,width=width,label="CUSUM Method")
plt.xticks(x + width,labels=traces)

for i in range(len(traces)):
    plt.text(DT1_x[i],detection_time_1[i], detection_time_1[i],va="bottom",ha="center",fontsize=8)
    plt.text(DT2_x[i],detection_time_2[i], detection_time_2[i],va="bottom",ha="center",fontsize=8) 

plt.xlabel('Dataset')
plt.ylabel('')
plt.title('Detection Time (s)')
plt.legend(loc="upper left")
plt.savefig(output_path+'detection_time')
plt.show()
plt.close()


td_1 =  [5.899999999999999, 1.2999999999999998, 0.8999999999999999, 0.7, 1.0999999999999999, 0.8999999999999999, 1.0999999999999999, 0.7]
bl_1 =  [0, 2, 5, 3, 2, 6, 1, 3]
td_2 =  [2.0, 0.2, 4.4, 0.0, 0.0, 0.4, 6.4, 0.0]
w_2 =   [17, 10, 8, 18, 9, 13, 2, 4]

td_1=np.around(np.array(td_1),2)
bl_1=np.around(np.array(bl_1),2)
td_2=np.around(np.array(td_2),2)
w_2=np.around(np.array(w_2),2)

x = np.arange(len(traces))
width = 0.2

plt.scatter(FP1_x,td_1,s=15,label="Threshold (Threshold Method)")
plt.scatter(FP1_x,bl_1,s=15,label="Burst Length (Threshold Method)")
plt.scatter(FP1_x,td_2,s=15,label="Threshold  (CUSUM Method)")
plt.scatter(FP1_x,w_2,s=15,label="Burst Length (CUSUM Method)")
plt.xticks(x + width,labels=traces)

for i in range(len(traces)):
    plt.text(FP1_x[i],td_1[i], td_1[i],va="bottom",ha="center",fontsize=8)
    plt.text(FP1_x[i],bl_1[i], bl_1[i],va="bottom",ha="center",fontsize=8)
    plt.text(FP1_x[i],td_2[i], td_2[i],va="bottom",ha="center",fontsize=8)
    plt.text(FP1_x[i],w_2[i], w_2[i],va="bottom",ha="center",fontsize=8) 

plt.xlabel('Dataset')
plt.ylabel('')
plt.title('Recommended Parameter Values')
plt.legend(loc="upper right")
plt.savefig(output_path+'Recommended_Values')
plt.show()
plt.close()
