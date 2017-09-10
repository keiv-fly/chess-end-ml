import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical

X = np.load("data\KPvK.npy")
X = np.swapaxes(X,1,2)
X = np.swapaxes(X,2,3)
df = pd.read_csv("data\KPvK_1_fen_wdl.csv")
y = to_categorical(df["wdl"]+2)

m = Sequential()
m.add(Conv2D(16, (3, 3), activation='relu', input_shape=(8,8,3), padding="same"))
m.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
m.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
m.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
m.add(Conv2D(16, (3, 3), activation='relu', padding="same"))
m.add(Flatten())
m.add(Dense(5, activation='softmax'))
adam=Adam()
m.compile(loss='categorical_crossentropy', optimizer=adam)
hist = m.fit(X, y, batch_size=32, epochs=10)
y_train = m.predict(X, batch_size=32, verbose=1)
y_train_cat=np.argmax(y_train,1)-2
acc = np.sum(y_train_cat==df["wdl"])/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist.history["loss"]]
m.summary()