import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model

X = np.load("data/KPvK.npy")
X = np.swapaxes(X,1,2)
X = np.swapaxes(X,2,3)
df = pd.read_csv("data/KPvK_1_fen_wdl.csv")
#y = to_categorical(df["wdl"]/2)
y = (df["wdl"]==2)*1
#extend matrix
X_orig = X
X_patt = np.zeros((X.shape[0],16,16,16),dtype=np.float64)
X[:,4:12,4:12,7:10]=X_orig


m = Sequential()
m.add(Conv2D(4, (1, 1), activation='relu', input_shape=(16,16,16), padding="same"))
m.add(Conv2D(4, (3, 3), activation='relu', padding="same"))
m.add(Conv2D(4, (3, 3), activation='relu', padding="same"))
m.add(Conv2D(4, (3, 3), activation='relu', padding="same"))
m.add(Conv2D(4, (3, 3), activation='relu', padding="same"))
m.add(Conv2D(4, (3, 3), activation='relu', padding="same"))
m.add(Conv2D(8, (1, 1), activation='relu', padding="same"))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(Flatten())
#m.add(Dense(32, activation='relu'))
m.add(Dense(1, activation='sigmoid'))
adam=Adam()
m.compile(loss='binary_crossentropy', optimizer=adam)
hist = m.fit(X, y, batch_size=32, epochs=1) #1024*8
y_train = m.predict(X, batch_size=1024*8, verbose=1)
y_train_cat=(y_train>0.5)*2
y_train_cat=y_train_cat.flatten()
acc = np.sum(y_train_cat==df["wdl"])/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist.history["loss"]]
m.summary()

hist2 = m.fit(X, y, batch_size=32, epochs=20)
y_train = m.predict(X, batch_size=1024*8, verbose=1)
y_train_cat=(y_train>0.5)*2
y_train_cat=y_train_cat.flatten()
acc = np.sum(y_train_cat==df["wdl"])/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist2.history["loss"]]
m.summary()
#m.save(r"data\model1KPvK.h5")
#m.save(r"data\model2KPvK.h5")
#m.save(r"data\model3KPvK_temp4.h5")
#m.save(r"data\model4KPvK_temp4.h5")
#m.save(r"data\model5KPvK_temp1.h5")
#m.save(r"data\model7KPvK_temp1.h5")
m=load_model(r"data\model7KPvK_temp1.h5")

#--------------------------------
nohup python py1.py>out.txt 2>&1 &
tail -f out.txt

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam , SGD
from keras.utils import to_categorical
from keras.models import load_model
import datetime

X = np.load("data/KPvK_w_5.npy")
X = np.swapaxes(X,1,2)
X = np.swapaxes(X,2,3)
df = pd.read_csv("data/KPvK_1_fen_wdl.csv")
y = (df["wdl"]==2)*1
m=load_model(r"data/model6KPvK_temp2.h5")
sgd = SGD(lr=0.001, momentum=0.9)
m.optimizer = sgd
#print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
hist2 = m.fit(X, y, batch_size=1024*8, epochs=5)
#datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
y_train = m.predict(X, batch_size=1024*8, verbose=1)
y_train_cat=(y_train>0.5)*2
y_train_cat=y_train_cat.flatten()
acc = np.sum(y_train_cat==df["wdl"])/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist2.history["loss"]]
m.summary()
m.save(r"data\model6KPvK_temp3.h5")