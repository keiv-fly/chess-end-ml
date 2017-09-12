import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model

X = np.load("data\KPvK.npy")
X = np.swapaxes(X,1,2)
X = np.swapaxes(X,2,3)
df = pd.read_csv("data\KPvK_1_fen_wdl.csv")
y = to_categorical(df["wdl"]/2)

m = Sequential()
m.add(Conv2D(6, (3, 3), activation='relu', input_shape=(8,8,3), padding="same"))
m.add(Conv2D(6, (3, 3), activation='relu', padding="same"))
m.add(Conv2D(6, (3, 3), activation='relu', padding="same"))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(Conv2D(6, (3, 3), activation='relu', padding="same"))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(Conv2D(6, (3, 3), activation='relu', padding="same"))
m.add(MaxPooling2D(pool_size=(2, 2)))
m.add(Flatten())
m.add(Dense(32, activation='relu'))
m.add(Dense(2, activation='softmax'))
adam=Adam()
m.compile(loss='categorical_crossentropy', optimizer=adam)
hist = m.fit(X, y, batch_size=32, epochs=10)
y_train = m.predict(X, batch_size=32, verbose=1)
y_train_cat=np.argmax(y_train,1)*2
acc = np.sum(y_train_cat==df["wdl"])/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist.history["loss"]]
m.summary()

hist2 = m.fit(X, y, batch_size=1024*8, epochs=1800)
y_train = m.predict(X, batch_size=1024*8, verbose=1)
y_train_cat=np.argmax(y_train,1)*2
acc = np.sum(y_train_cat==df["wdl"])/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist2.history["loss"]]
m.summary()
#m.save(r"data\model1KPvK.h5")
#m.save(r"data\model2KPvK.h5")
#m.save(r"data\model3KPvK_temp4.h5")
#m.save(r"data\model4KPvK_temp1.h5")
m=load_model(r"data\model4KPvK_temp1.h5")