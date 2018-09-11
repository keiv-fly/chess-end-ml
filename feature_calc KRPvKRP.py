import pandas as pd
import numpy as np
import datetime
import numba
from lib import features_from_table,generate_table


df=pd.read_hdf("data/KRPvKRP_table_5M_random_v2.h5", 'df1')

start_time = datetime.datetime.now()
X = features_from_table(df)
end_time = datetime.datetime.now()

np.save("data/KRPvKRP_16f_5M_random_v2.npy",X)
