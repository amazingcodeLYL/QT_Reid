import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
#折线图
# x = np.array([1,5,10])#点的横坐标
# k1 = np.array([0.8222,0.918,0.9344])#线1的纵坐标
# k2 = np.array([0.8988,0.9334,0.9435])#线2的纵坐标
# x_new = np.linspace(x.min(),x.max(),300)
# y_smooth = make_interp_spline(x, k1)(x_new)
# plt.plot(x,k1,'s-',color = 'r',label="ATT-RLSTM")#s-:方形
# plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
def smooth_xy(x_value: np.ndarray, y_value: np.ndarray):
    from scipy.interpolate import make_interp_spline
    model = make_interp_spline(x_value, y_value)
    x_smooth = np.linspace(x_value.min(), x_value.max(), 500)
    y_smooth = model(x_smooth)
    return x_smooth, y_smooth


# x_svd=np.array([1,5,10])
# y_svd=np.array([82.30,92.30,94.70])

# x_spreid=np.array([1,5,10])
# y_spreid=np.array([93.68,97.57,98.40])

# x_rerank=np.array([1,5,10])
# y_rerank=np.array([94.63,96.82,97.65])

# x_PIE=np.array([1,5,10])
# y_PIE=np.array([78.65,90.26,93.59])

# x_ours=np.array([1,5,10])
# y_ours=np.array([93.20,98.30,98.93])



# x_svd=np.array([1,5,10])
# y_svd=np.array([76.70,86.40,89.90])

# x_spreid=np.array([1,5,10])
# y_spreid=np.array([85.95,92.95,94.52])

# # x_rerank=np.array([1,5,10])
# # y_rerank=np.array([89.41,93.18,94.75])

# # x_PIE=np.array([1,5,10])
# # y_PIE=np.array([78.65,90.26,93.59])

# x_ours=np.array([1,5,10])
# y_ours=np.array([85.40,94.10,96.00])

x=np.array([1,5,10])
# y_Bas=np.array([73.10 , 92.70 , 96.70 ])
# y_Ver=np.array([84.60 ,97.60 ,98.90 ])
# y_M=np.array([88.20 ,98.20 ,99.10 ])
# y_QuM=np.array([75.53 ,95.15 ,99.16  ])
# y_Qu=np.array([74.47 ,96.92 ,98.95])
# y_Fu=np.array([74.21 ,94.33 ,97.54])
# y_Spi=np.array([88.50 ,97.80 ,98.60])
# y_Deep=np.array([85.40 ,97.60 ,99.40])
# y_PDC=np.array([88.70 ,98.61 ,99.24])
# y_Guo=np.array([87.50 ,97.85 ,99.45])
# y_MC=np.array([86.36 ,98.54 ,99.66])
# y_ours=np.array([89.95 ,98.67 ,99.89])


yMD=np.array([27.70 ,40.20 ,46.60  ])
yDM=np.array([50.60 ,63.80 ,70.90])
yMMS=np.array([6.90  ,12.00 ,15.20])
yMMAR=np.array([55.40 ,72.10 ,78.40])
yDMSMT=np.array([11.00 ,18.50 ,22.70])
yMDuke=np.array([53.70 ,69.30 ,75.60])




# xs,ys = smooth_xy(x,y)
# xs,ys = x, y
plt.grid(color='y',    
         linestyle='--',
         linewidth=1,
         alpha=0.3) 
plt.xlim((1, 10))
plt.plot(x,yMD,'o-',markersize='8',color = 'b',label="Market-1501→DukeMTMC-reID",linewidth=2)#o-:圆形
plt.plot(x,yDM,'s-',markersize='8',color = 'y',label="DukeMTMC-reID→Market-1501",linewidth=2)#o-:圆形
plt.plot(x,yMMS,'o-',markersize='8',color = 'gray',label="Market-1501→MSMT-17",linewidth=2)#o-:圆形
plt.plot(x,yMMAR,'s-',markersize='8',color = 'r',label="MSMT-17→Market-1501",linewidth=2)#o-:圆形
plt.plot(x,yDMSMT,'o-',markersize='8',color = 'm',label="DukeMTMC-reID→MSMT-17",linewidth=2)#o-:圆形
plt.plot(x,yMDuke,'s-',markersize='8',color = 'cyan',label="MSMT-17→DukeMTMC-reID",linewidth=2)#o-:圆形
# plt.plot(x,y_Spi,'o-',markersize='8',color = 'olive',label="Spindle",linewidth=2)#o-:圆形
# plt.plot(x,y_Deep,'s-',markersize='8',color = 'pink',label="DeepAlign",linewidth=2)#o-:圆形
# plt.plot(x,y_PDC,'o-',markersize='8',color = 'brown',label="PDC",linewidth=2)#o-:圆形
# plt.plot(x,y_Guo,'s-',markersize='8',color = 'purple',label="Guo et al. ",linewidth=2)#o-:圆形
# plt.plot(x,y_MC,'o-',markersize='8',color = 'orange',label="MC-PPMN (hnm)",linewidth=2)#o-:圆形
# plt.plot(x,y_ours,'s-',markersize='8',color = 'g',label="Ours",linewidth=2)#o-:圆形
plt.xlabel("rank")#横坐标名字
plt.ylabel("matching rate(%)")#纵坐标名字
plt.legend(loc = "best")#图例
plt.show()
