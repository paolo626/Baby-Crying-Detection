import os
import wave
import scipy.signal as signal
from scipy.fftpack import dct
import numpy as np
from sklearn.preprocessing import StandardScaler

def enframe(signal, nw, inc, winfunc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length=len(signal) #信号总长度
    if signal_length<=nw: #若信号长度小于一个帧的长度，则帧数定义为1
        nf=1
    else: #否则，计算帧的总长度
        nf=int(np.ceil((1.0*signal_length-nw+inc)/inc))
    pad_length=int((nf-1)*inc+nw) #所有帧加起来总的铺平后的长度
    zeros=np.zeros((pad_length-signal_length,)) #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal=np.concatenate((signal,zeros)) #填补后的信号记为pad_signal
    indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T  #相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices=np.array(indices,dtype=np.int32) #将indices转化为矩阵
    frames=pad_signal[indices] #得到帧信号
    win=np.tile(winfunc,(nf,1))  #window窗函数，这里默认取1
    return frames*win   #返回帧信号矩阵

def wavread(filename):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.short)  # 将字符串转化为short
    f.close()
    waveData = waveData * 1.0 / (max(abs(waveData+1)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    return framerate,waveData

def getData(filename,nw,inc):
    sample_rate,data = wavread(filename)
    # nw = 1024*8 #每一帧的长度
    # inc = 128*8 #帧移
    Frame = enframe(data[0], nw, inc,signal.hanning(nw))
    # Frame = enframe_noWindow(data[0], nw, inc)

    #time = np.arange(0, nw)

    # Frame = Frame / np.max(Frame)   # 归一化，标准化

    # plt.plot(time, Frame[0][:])
    # plt.xlabel("time (seconds)")
    # plt.ylabel("Amplitude")
    # plt.title("Left channel")
    # plt.grid()  # 标尺
    # plt.show()

    # 应用傅里叶变换
    NFFT = 512 # 2048#######
    mag_frames = np.absolute(np.fft.rfft(Frame, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

    # plt.figure(figsize=(20, 5))
    # plt.plot(pow_frames[0])
    # plt.grid()
    # plt.show()

    #Mel
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    #print(low_freq_mel, high_freq_mel)

    nfilt = 40
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    for i in range(1, nfilt + 1):
        left = int(bin[i-1])
        center = int(bin[i])
        right = int(bin[i+1])
        for j in range(left, center):
            fbank[i-1, j+1] = (j + 1 - bin[i-1]) / (bin[i] - bin[i-1])
        for j in range(center, right):
            fbank[i-1, j+1] = (bin[i+1] - (j + 1)) / (bin[i+1] - bin[i])
    #print(fbank.shape)


    # pow_frames = pow_frames[500:1000,:]
    #显示了全部的Filter bank
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB
    #print(filter_banks.shape)

######## MFCC
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]

    # sinusoidal liftering正弦提升
    cep_lifter = 23
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    print(mfcc.shape)

    return mfcc
########

######### Fbank
    # scaler = StandardScaler()
    # filter_banks_scaled = scaler.fit_transform(filter_banks.astype(np.float32).reshape(-1, 1)).reshape(-1, len(filter_banks), 40)
    #
    # return filter_banks_scaled[0]
#########

def create_label(data, application_id, power_threshold, dimension):
    labels = np.zeros((len(data),dimension))
    for i, value in enumerate(data):
        for j in application_id:
            labels[i,j] = 1
    return labels

chord_dict ={
    "awake": 0,
    "hug": 1,
    "hungry":2,
    "sleepy": 3,
    "uncomfortable": 4,
}



baby_len = 5

filepath = "wavs_all_cut/" #添加路径
dirnames= os.listdir(filepath) #得到文件夹下的所有文件名称

window = 20
wd_move = 10
#提取第一数据的data和label
sumData = getData(filepath+dirnames[0], 1024 * window, 128 * wd_move)


current_baby_names = dirnames[0][:-4]
current_baby_name_array = current_baby_names.split("_")[:-1]
current_chord_ids = np.zeros(len(current_baby_name_array))

for id, current_chord_name in enumerate(current_baby_name_array):
    current_chord_ids[id] = chord_dict[current_chord_name]
sumLabel = create_label(sumData, current_chord_ids.astype(np.int), 0, baby_len)


aaaa = 0
for fileName in dirnames[1:]:
    current_chord_datas = getData(filepath+fileName, 1024 * window, 128 * wd_move)
    sumData = np.append(sumData,current_chord_datas, axis=0)

    current_chord_names = fileName[:-4]
    current_chord_name_array = [current_chord_names.split("_")[0]]
    current_chord_ids = np.zeros(len(current_chord_name_array))
    for id, current_chord_name in enumerate(current_chord_name_array):
        current_chord_ids[id] = chord_dict[current_chord_name]
    sumLabel = np.append(sumLabel, create_label(current_chord_datas, current_chord_ids.astype(np.int), 0, baby_len), axis=0)
    print(aaaa)
    aaaa += 1


# 写入文件
np.savetxt(fname="FMCC_CSV/"+"wavs_old_cut_"+str(window)+"_"+str(wd_move)+"_float_Datas.csv", X=sumData,delimiter=",")
np.savetxt(fname="FMCC_CSV/"+"wavs_all_cut_"+str(window)+"_"+str(wd_move)+"_float_Labels.csv", X=sumLabel,delimiter=",")
# # 读取文件
# b = np.loadtxt(fname="FMCC_CSV/"+"wavs_old_cut_"+str(window)+"_"+str(wd_move)+"_Datas.csv", dtype=np.float, delimiter=",")
# d = np.loadtxt(fname="FMCC_CSV/"+"wavs_old_cut_"+str(window)+"_"+str(wd_move)+"_Labels.csv", dtype=np.float, delimiter=",")
# print(d)