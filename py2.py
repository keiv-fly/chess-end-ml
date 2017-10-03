import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
import datetime

X = np.load("data/KRPvKRP_7f.npy")
X = np.swapaxes(X,1,2)
X = np.swapaxes(X,2,3)
i_file=2
df=pd.read_hdf("data/KRPvKRP_table_" + '{:02d}'.format(i_file) + ".h5", 'df1')
y=np.zeros((len(df.index),5))
y[:,0] = (df["wdl"]==-2)*1
y[:,1] = (df["wdl"]==-1)*1
y[:,2] = (df["wdl"]==0)*1
y[:,3] = (df["wdl"]==1)*1
y[:,4] = (df["wdl"]==2)*1
m=load_model(r"data/model2KRPvKRP_temp2.h5")

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
hist = m.fit(X, y, batch_size=1024, epochs=100)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
y_train = m.predict(X, batch_size=1024*8, verbose=1)
y_train_cat=np.argmax(y_train,1)-2
acc = np.sum(y_train_cat==df.wdl)/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist.history["loss"]]
m.save(r"data\model2KRPvKRP_temp3.h5")
