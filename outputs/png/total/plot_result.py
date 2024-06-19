from matplotlib import pyplot as plt
import numpy as np

output_path="./outputs/png/total/"
traces = []
for i in range(1,9):
    traces.append("trace"+str(i))
FP_rate_1 =         [0.1683521713265913, 0.32088148347218487, 0.26820866141732286, 0.09125596184419714, 0.10752688172043011, 0.6508972267536705, 0.27411944869831545, 0.2575687005123428]
detection_time_1 =  [0.0, 0.1, 0.4, 0.3, 0.2, 0.4, 10.195, 0.7]
FP_rate_2 =         [0.009220701963117191, 0.0, 0.011811023622047244, 0.0, 0.0, 0.0, 0.0, 0.0]
detection_time_2 =  [117.077, 166.67, 82.46499999999999, 173.153, 215.462, 46.94, 160.129, 183.374]

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