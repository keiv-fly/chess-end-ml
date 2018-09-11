import keras
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, MaxPool2D, Dropout
from keras.layers.merge import Add
from keras.layers import Conv2D, Conv3D, GlobalMaxPool2D, MaxPooling3D, MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers.core import Reshape, Permute
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

X = np.load("data/KRPvKRP_16f_5M_random_v2.npy")
X = np.swapaxes(X,1,2)
X = np.swapaxes(X,2,3)

X = X[:100000,:,:,:]

df=pd.read_hdf("data/KRPvKRP_table_5M_random_v2.h5", 'df1')
df = df[:100000]
y=np.zeros((len(df.index),5))
y[:,0] = (df["wdl"]==-2)*1
y[:,1] = (df["wdl"]==-1)*1
y[:,2] = (df["wdl"]==0)*1
y[:,3] = (df["wdl"]==1)*1
y[:,4] = (df["wdl"]==2)*1

X2 = np.load("data/KRPvKRP_16f_10K_random_test_v2.npy")
X2 = np.swapaxes(X2,1,2)
X2 = np.swapaxes(X2,2,3)

df2=pd.read_hdf("data/KRPvKRP_table_10K_random_test_v2.h5", 'df1')
y2=np.zeros((len(df2.index),5))
y2[:,0] = (df2["wdl"]==-2)*1
y2[:,1] = (df2["wdl"]==-1)*1
y2[:,2] = (df2["wdl"]==0)*1
y2[:,3] = (df2["wdl"]==1)*1
y2[:,4] = (df2["wdl"]==2)*1

inputs = Input(shape=(8,8,16))
x = inputs
#drp = 0.1
base_channels=32
#x1 = x
x = Conv2D(base_channels, kernel_size = 3, padding="same")(x)
x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
x1 = x
#x = BatchNormalization()(x)
#x = Activation(activation='relu')(x)
x = Conv2D(base_channels, kernel_size = 3, padding="same")(x)
#x =Add()([x,x1])
#x1 = x
x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
x = Conv2D(base_channels, kernel_size = 3, padding="same")(x)
x =Add()([x,x1])

x1 = x
x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
x = Conv2D(base_channels, kernel_size = 3, padding="same")(x)
#x =Add()([x,x1])
#x1 = x
x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
x = Conv2D(base_channels, kernel_size = 3, padding="same")(x)
x =Add()([x,x1])

x1 = x
x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
x = Conv2D(base_channels, kernel_size = 3, padding="same")(x)
#x =Add()([x,x1])
#x1 = x
x = BatchNormalization()(x)
x = Activation(activation='relu')(x)
x = Conv2D(base_channels, kernel_size = 3, padding="same")(x)
x =Add()([x,x1])

x = MaxPool2D()(x)

x = MaxPool2D()(x)

x = MaxPool2D()(x)

#x = Dropout(drp)(x)

x = Flatten()(x)
x = Dense(base_channels*8, activation='relu')(x)
#x = Dropout(drp)(x)
predictions = (Dense(5, activation='softmax'))(x)
adam=Adam()

model = Model(inputs=inputs, outputs=predictions)
#sgd = SGD(lr=0.005, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['acc'])
hist = model.fit(X, y, batch_size=256, epochs=1, validation_data=(X2,y2)) #1024*8
model.save(r"data/model20KRPvKRP.h5")

df_hist = pd.DataFrame(list(zip(hist.history["loss"],hist.history["val_loss"])))
df_hist.columns = ["loss", "val_loss"]
df_hist.to_csv("data/model20KRPvKRP_r_acc.csv", index=False)
# model.summary()

#y_train = model.predict(X2, batch_size=1024*2, verbose=1)
#y_train_cat=np.argmax(y_train,1)-2
#acc = np.sum(y_train_cat==df2.wdl)/y_train_cat.shape[0]
#print("\nerror = ", 1-acc)

#model.save(r"data/model2KRPvKRP_temp1.h5")
#model=load_model(r"data/model1KRPvKRP_temp1.h5")