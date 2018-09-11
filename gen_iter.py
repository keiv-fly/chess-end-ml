from lib import generate_table
from pathlib import Path
import datetime
import pandas
import os

start_time = datetime.datetime.now()
df= generate_table(1000000)
end_time = datetime.datetime.now()
print("Position generation time: ", end_time - start_time)

filename = r"data/KRPvKRP_table_10M_random_iter.h5"
my_file = Path(filename)
if my_file.exists():
    os.remove(filename)
df.to_hdf(filename, 'df1', complib='blosc:lz4', complevel=9)
