import matplotlib.pyplot as plt
import numpy as np

# x = np.array([68.7,72.6,76.6,82.3,75.3,81.3,69.0,73.9,77.7,81.6,69.1,75.7,84.7])
# y = np.array([88.1,89.4,91.4,93.1,90.0,92.5,87.7,89.9,90.6,93.8,84.9,89.5,93.2])

x = np.array([91.00,87.90,82.00,92.70,91.70,82.50,85.00,92.00])
y = np.array([88.00,93.00,88.00,86.80,83.40,84.10,91.00,95.00])
txt = ['Zhu et al.','P-GAN','YOLOv3','Wang et al.','Lu et al.','MSA-YOLOv3','YOLOv4','Ours']

import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# print(x[:12])
plt.scatter(x[:7], y[:7],marker='o', c='g')
plt.scatter(x[7], y[7],marker='o', c='r')
print(len(x))
for i in range(len(x)):
    plt.annotate(txt[i], xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1)) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
plt.xlabel("Precision(%)")#横坐标名字
plt.ylabel("Recall(%)")#纵坐标名字
plt.show()