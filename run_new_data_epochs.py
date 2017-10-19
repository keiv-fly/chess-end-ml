# nohup python -u run_new_data_epochs.py > out.txt 2>&1 &
# tail -f out.txt

print("imports")
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import load_model
import datetime
import os
from lib import features_from_table,generate_table

print("code start")

# load validation feature boards
X2 = np.load("data/KRPvKRP_16f_10K_random_test_v2_02.npy")
X2 = np.swapaxes(X2,1,2)
X2 = np.swapaxes(X2,2,3)

# load validation result array
df2=pd.read_hdf("data/KRPvKRP_table_10K_random_test_v2_02.h5", 'df1')
y2=np.zeros((len(df2.index),5))
y2[:,0] = (df2["wdl"]==-2)*1
y2[:,1] = (df2["wdl"]==-1)*1
y2[:,2] = (df2["wdl"]==0)*1
y2[:,3] = (df2["wdl"]==1)*1
y2[:,4] = (df2["wdl"]==2)*1

# load the previous model
m=load_model(r"data/model3KRPvKRP_r_temp9.h5")

# change the learning rate
adam=Adam()
#sgd = SGD(lr=0.002, momentum=0.9)
m.compile(loss='categorical_crossentropy', optimizer=adam)

hist_loss=[]
hist_val_loss=[]
hist_acc=[]

for i_data_epochs in range(60):
    print("\niteration number: ",i_data_epochs) 
    # generate table with positions and results (wdl) in parallel to calculation
    os.system("nohup python -u gen_iter.py > gen_out.txt 2>&1 &")

    #load table calculated in the previous iteration
    df = pd.read_hdf("data/KRPvKRP_table_10M_random_iter.h5")

    # generate feature boards from the table above
    start_time = datetime.datetime.now()
    X=features_from_table(df)
    end_time = datetime.datetime.now()
    print("Feature boards generation time: ",end_time - start_time)

    # swap axis for Keras
    X = np.swapaxes(X,1,2)
    X = np.swapaxes(X,2,3)

    # generate 5 arrays of results for each wdl (-2,-1,0,1,2)
    y=np.zeros((len(df.index),5))
    y[:,0] = (df["wdl"]==-2)*1
    y[:,1] = (df["wdl"]==-1)*1
    y[:,2] = (df["wdl"]==0)*1
    y[:,3] = (df["wdl"]==1)*1
    y[:,4] = (df["wdl"]==2)*1



    # calculate epochs and calculate execution time
    start_time = datetime.datetime.now()
    print(start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
    hist = m.fit(X, y, batch_size=256, epochs=1, validation_data=(X2,y2))
    end_time = datetime.datetime.now()
    os.remove(r"data/model3KRPvKRP_r_temp9.h5")
    m.save(r"data/model3KRPvKRP_r_temp9.h5")
    print(end_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
    print("Training time: ",end_time - start_time)

    # ouput training and test loss
    _ = [print(x) for x in hist.history["loss"]]
    print()
    _ = [print(x) for x in hist.history["val_loss"]]
    hist_loss = hist_loss + hist.history["loss"]
    hist_val_loss = hist_val_loss + hist.history["val_loss"]


    # calculate and print accuracy
    y_train = m.predict(X2, batch_size=1024*2, verbose=1)
    y_train_cat=np.argmax(y_train,1)-2
    acc = np.sum(y_train_cat==df2.wdl)/y_train_cat.shape[0]
    hist_acc.append(acc)
    n_errors = y_train_cat.shape[0] - np.sum(y_train_cat==df2.wdl)
    print("\n\naccuracy\n% errors\n# errors")
    print(acc)
    print(1-acc)
    print(n_errors)

_ = [print(x) for x in hist_loss]
print()
_ = [print(x) for x in hist_val_loss]
print()
_ = [print(x) for x in hist_acc]
