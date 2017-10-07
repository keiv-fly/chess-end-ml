import keras
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers.merge import Add
from keras.layers import Conv2D, Conv3D, GlobalMaxPool2D, MaxPooling3D, MaxPooling2D
from keras.layers import SeparableConv2D
from keras.layers.core import Reshape, Permute
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

X = np.load("data/KRPvKRP_17f_5M_random.npy")
X = np.swapaxes(X,1,2)
X = np.swapaxes(X,2,3)

X = X[:10240,:,:,:]

i_file=2
df=pd.read_hdf("data/KRPvKRP_table_5M_random.h5", 'df1')
df = df[:10240]
y=np.zeros((len(df.index),5))
y[:,0] = (df["wdl"]==-2)*1
y[:,1] = (df["wdl"]==-1)*1
y[:,2] = (df["wdl"]==0)*1
y[:,3] = (df["wdl"]==1)*1
y[:,4] = (df["wdl"]==2)*1

inputs = Input(shape=(8,8,17))
x = inputs
#x = BatchNormalization()(inputs)
x1 = Conv2D(32, (1, 1), padding="same")(x)
x2 = Conv2D(32, (5, 5), padding="same")(x)
x = Add()([x1, x2])
x = Activation("relu")(x)

def res_layer(x):
    x1 = Conv2D(32, (1, 1), padding="same")(x)
    x = Add()([x1, x])
    x = Activation("relu")(x)
    return x
for i in range(12):
    x = res_layer(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
# x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
#x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
#x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
#x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
#x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
#x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)
#x = Conv2D(32, (3, 3), activation='relu', padding="same")(x)

x = Flatten()(x)
predictions = (Dense(5, activation='softmax'))(x)
#adam=Adam()

model = Model(inputs=inputs, outputs=predictions)
sgd = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
hist = model.fit(X, y, batch_size=256, epochs=1) #1024*8
model.save(r"data/model1KRPvKRP_r_temp1.h5")
y_train = model.predict(X, batch_size=1024*8, verbose=1)
y_train_cat=np.argmax(y_train,1)-2
acc = np.sum(y_train_cat==df.wdl)/y_train_cat.shape[0]
print("\naccuracy = ", acc)
_ = [print(x) for x in hist.history["loss"]]
model.summary()

#model.save(r"data/model2KRPvKRP_temp1.h5")
#model=load_model(r"data/model1KRPvKRP_temp1.h5")