import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
import datetime

X = np.load("data/KPvK_w_5.npy")
X = np.swapaxes(X,1,2)
X = np.swapaxes(X,2,3)
df = pd.read_csv("data/KPvK_1_fen_wdl.csv")
y = (df["wdl"]==2)*1
m=load_model(r"data/model87KPvK_temp11.h5")

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
hist2 = m.fit(X, y, batch_size=163328, epochs=120)
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
y_train = m.predict(X, batch_size=256, verbose=1)
y_train_cat=(y_train>0.5)*2
y_train_cat=y_train_cat.flatten()
acc = np.sum(y_train_cat==df["wdl"])/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist2.history["loss"]]
m.summary()
m.save(r"data\model87KPvK_temp11.h5")
