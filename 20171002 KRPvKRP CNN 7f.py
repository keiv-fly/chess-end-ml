import keras
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Conv2D, Conv3D, GlobalMaxPool2D, MaxPooling3D, MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers.core import Reshape, Permute
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

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

inputs = Input(shape=(8,8,7))
x = Conv2D(8, (1, 1), activation='relu', padding="same")(inputs)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), activation='relu', padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), activation='relu', padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), activation='relu', padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), activation='relu', padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), activation='relu', padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), activation='relu', padding="same")(x)
x = BatchNormalization()(x)
x = Permute((1,3,2))(x)
x=Conv2D(8, (1, 1), activation='relu', padding="same")(x)
x = Permute((3, 2, 1))(x)
x = Conv2D(8, (1, 1), activation='relu', padding="same")(x)
x = Permute((3, 1, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(16, (1, 1), activation='relu', padding="same")(x)
x = GlobalMaxPool2D()(x)
predictions = (Dense(5, activation='softmax'))(x)
adam=Adam()
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer=adam)
hist = model.fit(X, y, batch_size=256, epochs=4) #1024*8
model.save(r"data/model2KRPvKRP_temp2.h5")
y_train = model.predict(X, batch_size=1024*8, verbose=1)
y_train_cat=np.argmax(y_train,1)-2
acc = np.sum(y_train_cat==df.wdl)/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist.history["loss"]]
model.summary()

#model.save(r"data/model2KRPvKRP_temp1.h5")
#model=load_model(r"data/model1KRPvKRP_temp1.h5")