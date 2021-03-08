import os
import wave
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.callbacks as kcallbacks


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
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
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

    time = np.arange(0, nw)

    # Frame = Frame / np.max(Frame)   # 归一化，标准化

    # plt.plot(time, Frame[0][:])
    # plt.xlabel("time (seconds)")
    # plt.ylabel("Amplitude")
    # plt.title("Left channel")
    # plt.grid()  # 标尺
    # plt.show()

    # 应用傅里叶变换
    NFFT = 512
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
    #print(fbank)


    # pow_frames = pow_frames[500:1000,:]
    #显示了全部的Filter bank
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB
    #print(filter_banks.shape)

    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)]

    # sinusoidal liftering正弦提升
    cep_lifter = 23
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    return mfcc

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

# filepath = "wavs_old_cut/" #添加路径
# dirnames= os.listdir(filepath) #得到文件夹下的所有文件名称

# 读取文件
# d4 = np.loadtxt(fname="Mel_CSV/wavs_old_cut_8_6_Datas.csv", dtype=np.float, delimiter=",")
# l4 = np.loadtxt(fname="Mel_CSV/wavs_old_cut_8_6_Labels.csv", dtype=np.float, delimiter=",")
# d6 = np.loadtxt(fname="Mel_CSV/wavs_old_cut_8_7_Datas.csv", dtype=np.float, delimiter=",")
# l6 = np.loadtxt(fname="Mel_CSV/wavs_old_cut_8_7_Labels.csv", dtype=np.float, delimiter=",")
# d8 = np.loadtxt(fname="Mel_CSV/wavs_old_cut_8_8_Datas.csv", dtype=np.float, delimiter=",")
# l8 = np.loadtxt(fname="Mel_CSV/wavs_old_cut_8_8_Labels.csv", dtype=np.float, delimiter=",")
d10 = np.loadtxt(fname="FMCC_CSV/wavs_old_cut_20_10_float_Datas.csv", dtype=np.float, delimiter=",")
l10 = np.loadtxt(fname="FMCC_CSV/wavs_all_cut_20_10_float_Labels.csv", dtype=np.float, delimiter=",")

sumData =d10
sumLabel =l10
print(sumData.shape)
print(sumLabel.shape)




lookback = 10
step = 1
delay = 0
batch_size = 128

def generator(data, label, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=10):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),baby_len))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = label[rows[j] + delay]
        
        #print("samples:",samples.shape)
        #print(targets.shape)
        # samples = samples.reshape((batch_size,10,12,1))
        yield samples, targets


train_gen = generator(sumData,
                      sumLabel,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=len(sumLabel),
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
#
# val_gen = generator(val_sumData,
#                       val_sumLabel,
#                       lookback=lookback,
#                       delay=delay,
#                       min_index=0,
#                       max_index=len(val_sumLabel),
#                       step=step,
#                       batch_size=batch_size)

train_steps = len(sumLabel) // batch_size
# val_steps = (len(val_sumLabel) -lookback) // batch_size

model = Sequential()
model.add(layers.Conv1D(64, 4, padding="same", activation='relu', input_shape=(None,sumData.shape[-1])))
# model.add(layers.Conv1D(128, 2, padding="same", activation='relu'))
# model.add(layers.MaxPooling1D(2))
model.add(layers.Bidirectional(
    layers.LSTM(64,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
model.add(layers.Bidirectional(
    layers.LSTM(128,dropout=0.2,recurrent_dropout=0.1)))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(baby_len,activation='softmax'))
model.summary()

model.compile(optimizer=RMSprop(), loss='categorical_crossentropy',metrics=['acc'])

#
# earlyStopping = kcallbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
# saveBestModel = kcallbacks.ModelCheckpoint("Model/BabyNew_normal3.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(train_gen,
                              steps_per_epoch=train_steps,
                              epochs=16)


loss = history.history['loss']
# val_loss = history.history['val_loss'] #

epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
# plt.legend()
plt.show()

plt.clf() #清空图表

acc_values = history.history['acc']
# val_acc_values  = history.history['val_acc']

plt.plot(epochs,acc_values,'bo',label='Training acc') #bo是蓝色圆点
# plt.plot(epochs,val_acc_values,'b',label='Validation acc') #b是蓝色实线
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

model.save('Model/FMCC_Baby_noval_20_8_oldcut_16epo.h5')