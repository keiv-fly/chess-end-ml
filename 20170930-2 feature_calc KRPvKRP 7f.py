import pandas as pd
import numpy as np
import datetime
import numba

i_file=2
df=pd.read_hdf("data/KRPvKRP_table_" + '{:02d}'.format(i_file) + ".h5", 'df1')

#df=df.head(1000000)




board_shape = (8, 8)

n_rows = len(df.index)

data = np.zeros((n_rows, 7, 8, 8), dtype=np.float16)
ii=0
print(datetime.datetime.now().strftime("%H:%M:%S.%f"))
print(ii)
ii = ii+1
data[:,0, :, :] = df.move.values[:, np.newaxis, np.newaxis]
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows/1000))):
    m_i = min(1000*(i+1),n_rows)
    idx = list(range(i*1000,m_i))
    data[idx,1,df.K[idx] // 8, df.K[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows/1000))):
    m_i = min(1000*(i+1),n_rows)
    idx = list(range(i*1000,m_i))
    data[idx,2,df.k[idx] // 8, df.k[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows / 1000))):
    m_i = min(1000 * (i + 1), n_rows)
    idx = list(range(i * 1000, m_i))
    data[idx, 3, df.R[idx] // 8, df.R[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows / 1000))):
    m_i = min(1000 * (i + 1), n_rows)
    idx = list(range(i * 1000, m_i))
    data[idx, 4, df.r[idx] // 8, df.r[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows / 1000))):
    m_i = min(1000 * (i + 1), n_rows)
    idx = list(range(i * 1000, m_i))
    data[idx, 5, df.P[idx] // 8, df.P[idx] % 8] = 1
print(ii)
ii = ii+1
for i in range(int(np.ceil(n_rows / 1000))):
    m_i = min(1000 * (i + 1), n_rows)
    idx = list(range(i * 1000, m_i))
    data[idx, 6, df.p[idx] // 8, df.p[idx] % 8] = 1
print(ii)
ii = ii+1
print(datetime.datetime.now().strftime("%H:%M:%S.%f"))

np.save("data/KRPvKRP_7f.npy",data)