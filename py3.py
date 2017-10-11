#nohup python -u py3.py > out.txt 2>&1 &
# tail -f out.txt

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import load_model
import datetime

#generate Numpy Data from a table
df=pd.read_hdf("data/KRPvKRP_table_10M_random_v2_03.h5", 'df1')

# df=df[5000000:5010240].reset_index(drop=True)

board_shape = (8, 8)

n_rows = len(df.index)

data = np.zeros((n_rows, 16, 8, 8), dtype=np.float16)
ii=0
print(datetime.datetime.now().strftime("%H:%M:%S.%f"))
print(ii)
ii = ii+1
# data[:,0, :, :] = df.move.values[:, np.newaxis, np.newaxis]
# print(ii)
# ii = ii+1
for i in range(int(np.ceil(n_rows/1000))):
    m_i = min(1000*(i+1),n_rows)
    idx = list(range(i*1000,m_i))
    data[idx,0,df.K[idx].values // 8, df.K[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows/1000))):
    m_i = min(1000*(i+1),n_rows)
    idx = list(range(i*1000,m_i))
    data[idx,1,df.k[idx] // 8, df.k[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows / 1000))):
    m_i = min(1000 * (i + 1), n_rows)
    idx = list(range(i * 1000, m_i))
    data[idx, 2, df.R[idx] // 8, df.R[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows / 1000))):
    m_i = min(1000 * (i + 1), n_rows)
    idx = list(range(i * 1000, m_i))
    data[idx, 3, df.r[idx] // 8, df.r[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows / 1000))):
    m_i = min(1000 * (i + 1), n_rows)
    idx = list(range(i * 1000, m_i))
    data[idx, 4, df.P[idx] // 8, df.P[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows / 1000))):
    m_i = min(1000 * (i + 1), n_rows)
    idx = list(range(i * 1000, m_i))
    data[idx, 5, df.p[idx] // 8, df.p[idx] % 8] = 1
print(ii)
ii = ii+1
data[:,6,:, :] = ((df.K // 8) / 7)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
data[:,7,:, :] = ((df.k // 8) / 7)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
data[:,8,:, :] = ((df.R // 8) / 7)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
data[:,9,:, :] = ((df.r // 8) / 7)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
data[:,10,:, :] = ((df.P // 8) / 7)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
data[:,11,:, :] = ((df.p // 8) / 7)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
data[:,12,:, :] = (np.maximum(abs((df.P // 8) - (df.K // 8)), abs((df.P % 8) - (df.K % 8))) / 8)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
data[:,13,:, :] = (np.maximum(abs((df.P // 8) - (df.k // 8)), abs((df.P % 8) - (df.k % 8))) / 8)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
data[:,14,:, :] = (np.maximum(abs((df.p // 8) - (df.K // 8)), abs((df.P % 8) - (df.K % 8))) / 8)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
data[:,15,:, :] = (np.maximum(abs((df.p // 8) - (df.k // 8)), abs((df.P % 8) - (df.k % 8))) / 8)[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
print(datetime.datetime.now().strftime("%H:%M:%S.%f"))

#------------------------------------------------------------------------------

X=data

#X = np.load("data/KRPvKRP_16f_5M_random_v2_02.npy")
X = np.swapaxes(X,1,2)
X = np.swapaxes(X,2,3)

#X = X[:10240,:,:,:]

#Table is already read
#df=pd.read_hdf("data/KRPvKRP_table_10M_random_v2_03.h5", 'df1')
#df = df[:10240]
y=np.zeros((len(df.index),5))
y[:,0] = (df["wdl"]==-2)*1
y[:,1] = (df["wdl"]==-1)*1
y[:,2] = (df["wdl"]==0)*1
y[:,3] = (df["wdl"]==1)*1
y[:,4] = (df["wdl"]==2)*1

X2 = np.load("data/KRPvKRP_16f_10K_random_test_v2_02.npy")
X2 = np.swapaxes(X2,1,2)
X2 = np.swapaxes(X2,2,3)

df2=pd.read_hdf("data/KRPvKRP_table_10K_random_test_v2_02.h5", 'df1')
y2=np.zeros((len(df2.index),5))
y2[:,0] = (df2["wdl"]==-2)*1
y2[:,1] = (df2["wdl"]==-1)*1
y2[:,2] = (df2["wdl"]==0)*1
y2[:,3] = (df2["wdl"]==1)*1
y2[:,4] = (df2["wdl"]==2)*1


m=load_model(r"data/model3KRPvKRP_r_temp5.h5")

sgd = SGD(lr=0.002, momentum=0.9)
m.compile(loss='categorical_crossentropy', optimizer=sgd)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
hist = m.fit(X, y, batch_size=256, epochs=24, validation_data=(X2,y2))
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
m.save(r"data/model3KRPvKRP_r_temp6.h5")

_ = [print(x) for x in hist.history["loss"]]
print()
_ = [print(x) for x in hist.history["val_loss"]]
#df_hist = pd.DataFrame(list(zip(hist.history["loss"],hist.history["val_loss"])))
#df_hist.columns = ["loss", "val_loss"]
#df_hist.to_csv("data/model3KRPvKRP_r_temp5_acc.csv", index=False)
# model.summary()

y_train = m.predict(X2, batch_size=1024*2, verbose=1)
y_train_cat=np.argmax(y_train,1)-2
acc = np.sum(y_train_cat==df2.wdl)/y_train_cat.shape[0]
print("\naccuracy = ", acc)
