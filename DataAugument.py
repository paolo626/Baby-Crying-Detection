import random
import numpy as np
import math

### //ddd
from scipy.fftpack import fft
class SpecAugment():
    def __init__(self, framesamplerate = 16000, timewindow = 25, timeshift = 10):
        self.time_window = timewindow
        self.window_length = int(framesamplerate / 1000 * self.time_window) # 计算窗长度的公式，目前全部为400固定值
        self.x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * (self.x) / (400 - 1) ) # 汉明窗
    
    def run(self, wavsignal, fs = 16000):
        if(16000 != fs):
            raise ValueError('[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')
        
        # wav波形 加时间窗以及时移10ms
        time_window = 25 # 单位ms
        window_length = int(fs / 1000 * time_window) # 计算窗长度的公式，目前全部为400固定值
        
        wav_arr = np.array(wavsignal)
        #wav_length = len(wavsignal[0])
        wav_length = wav_arr.shape[1]
        
        range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
        data_input = np.zeros((range0_end, window_length // 2), dtype = np.float) # 用于存放最终的频率特征数据
        data_line = np.zeros((1, window_length), dtype = np.float)
        
        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + 400
            
            data_line = wav_arr[0, p_start:p_end]
            
            data_line = data_line * self.w # 加窗
            
            #data_line = np.abs(fft(data_line)) / wav_length
            data_line = np.abs(fft(data_line))
            
            data_input[i]=data_line[0: window_length // 2] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
            
        #print(data_input.shape)
        data_input = np.log(data_input + 1)
        # 开始对得到的特征应用SpecAugment
        mode = random.randint(1,100)
        h_start = random.randint(1,data_input.shape[0])
        h_width = random.randint(1,100)
        v_start = random.randint(1,data_input.shape[1])
        v_width = random.randint(1,100)
        if(mode <= 60): # 正常特征 60%
            pass
        elif(mode > 60 and mode <=75): # 横向遮盖 15%
            data_input[h_start:h_start+h_width,:] = 0
            pass
        elif(mode > 75 and mode <= 90): # 纵向遮盖 15%
            data_input[:,v_start:v_start+v_width] = 0
            pass
        else: # 两种遮盖叠加 10%
            data_input[h_start:h_start+h_width,:v_start:v_start+v_width] = 0
            pass
        
        return data_input

        

feat_sa = SpecAugment()
data_input = feat_sa.run(wavsignal, fs)