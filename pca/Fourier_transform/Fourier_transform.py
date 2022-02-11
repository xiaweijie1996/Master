from scipy.fftpack import fft,ifft
import numpy as np
import copy
import sys
sys.path.append(r'../')
from Data_process_toolkit import *
# import matplotlib.pyplot as plt
# from matplotlib.pylab import mpl
# import pickle
# load_data = open(r'../data_after_process_weekday.pickle','rb')
# data_weekday = pickle.load(load_data)

def FFT(data,n=0,redata=False):
    
    """
    Input: a series of data, number of repeated times
    Output: 
    repeated data,
    comples number,
    length of complex number, 振幅
    相位
    """
    
    "reinforce the data"
    re_data1 = copy.deepcopy(data)
    re_data = copy.deepcopy(data)
    for i in range(n):
        re_data = np.hstack((re_data1,re_data))
    fft_y = fft(re_data)
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)              
    angle_y = np.angle(fft_y) #取复数的角度
    fft_y[0] = 0
    abs_y[0] = 0 
    l=int(len(fft_y)/2)  
    if redata == False:   
        return  fft_y
    return fft_y,re_data

def FFT_dicit(input_dicit,z,n=0):
    """
    INput: Dicitoinary, z represent the index of colums in matrix
    """
    fre = 1
    out_dicit_fft = {}
    out_dicit_fft_h = {}
    for i in range(len(input_dicit)):
        out_dicit_fft[i] = FFT(input_dicit[i][:,z],n)
    for i in range(len(out_dicit_fft)):
        # z = np.empty((0,2*len(out_dicit_fft[i])))
        abs_out_data = np.abs(out_dicit_fft[i])[int(len(out_dicit_fft[i])*fre/2):]
        angle_out_data = np.angle(out_dicit_fft[i])[int(len(out_dicit_fft[i])*fre/2):]
        z_v = np.vstack((abs_out_data,angle_out_data))
        z_h = np.hstack((stand_normal_reverse(abs_out_data).normal(),
                          stand_normal_reverse(angle_out_data).normal()))
        # z_h = stand_normal_reverse(abs_out_data).normal()
        out_dicit_fft[i] = z_v
        out_dicit_fft_h[i] = z_h
    return out_dicit_fft,out_dicit_fft_h

# a,b,c,d,l=FFT(data_weekday[0][:,1],10)
# x=np.array(range(a))

# x=np.linspace(0, 1,l)
# x1=np.linspace(0, 1,l*2)

# mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
# mpl.rcParams['axes.unicode_minus']=False       #显示负号

# plt.title('原始波形')
# plt.plot(x1,a)   
# plt.show()

# plt.title('双边振幅谱(未求振幅绝对值)',color='black') 
# plt.plot(x,b,'black')
# plt.show()

# plt.plot(x,c,'r')
# plt.title('双边振幅谱(未归一化)',fontsize=9,color='red') 
# plt.show()

# plt.plot(x,d,'violet')
# plt.title('双边相位谱(未归一化)',fontsize=9,color='violet')
# plt.show()