import pandas as pd
import numpy as np
import datetime
import numba


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

np.save("data/KRPvKRP_16f_10M_random_v2_03.npy",data)
