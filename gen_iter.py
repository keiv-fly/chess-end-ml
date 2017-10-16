from lib import generate_table
import datetime
import pandas
import os

start_time = datetime.datetime.now()
df= generate_table()
end_time = datetime.datetime.now()
print("Position generation time: ", end_time - start_time)

os.remove(r"data/KRPvKRP_table_10M_random_iter.h5")
df.to_hdf("data/KRPvKRP_table_10M_random_iter.h5", 'df1', complib='blosc:lz4', complevel=9)
