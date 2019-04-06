# nohup /bin/bash run.sh > out.txt 2>&1 &
# tail -f out.txt

# nohup python run_one.py > out.txt 2>&1 &
# python -u run_one.py

print("----------------------------")
print("\nNew iteration\n")
print("imports")
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import CSVLogger
import datetime
import os
from lib import features_from_table,generate_table
import signal, sys


print(datetime.datetime.now().strftime("%H:%M:%S.%f"))
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
m=load_model(r"data/model20KRPvKRP.h5")

# change the learning rate
#adam=Adam()
sgd = SGD(lr=0.0001, momentum=0.9)
#m.compile(loss='categorical_crossentropy', optimizer=adam) #sgd) #adam)

acc_hist = []
val_acc_hist = []
loss_hist = []
val_loss_hist = []

def exit_gracefully(signum, frame):
    print("\n\nAll Python Iterations\nloss\nval_loss\nacc\nval_acc")
    _ = [print(x) for x in loss_hist]
    print()
    _ = [print(x) for x in val_loss_hist]
    print()
    _ = [print(x) for x in acc_hist]
    print()
    _ = [print(x) for x in val_acc_hist]
    print()
    sys.exit()


def print_current_results(signum, frame):
    print("\n\nAll Python Iterations\nloss\nval_loss\nacc\nval_acc")
    _ = [print(x) for x in loss_hist]
    print()
    _ = [print(x) for x in val_loss_hist]
    print()
    _ = [print(x) for x in acc_hist]
    print()
    _ = [print(x) for x in val_acc_hist]
    print()

signal.signal(signal.SIGINT, exit_gracefully)
signal.signal(signal.SIGTERM, exit_gracefully)
#signal.signal(signal.SIGUSR1, print_current_results)

for i_data_epochs in range(10000):
    print("\nPython iteration number: ", i_data_epochs)
    # generate table with positions and results (wdl) in parallel to calculation
    #os.system("nohup python -u gen_iter.py > gen_out.txt 2>&1 &")
    os.system("START python -u gen_iter.py > gen_out.txt")

    #load table calculated in the previous iteration
    df = pd.read_hdf("data/KRPvKRP_table_10M_random_iter.h5")
    #if i_data_epochs == 0:
        #df = df[:500000]
    # generate feature boards from the table above
    start_time = datetime.datetime.now()
    X=features_from_table(df)
    end_time = datetime.datetime.now()
    #print("Feature boards generation time: ",end_time - start_time)

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

    csv_logger = CSVLogger('training.log', append=True)

    # calculate epochs and calculate execution time
    start_time = datetime.datetime.now()
    #print(start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
    hist = m.fit(X, y, batch_size=256, epochs=1, validation_data=(X2,y2), callbacks=[csv_logger])
    end_time = datetime.datetime.now()
    os.remove(r"data/model20KRPvKRP.h5")
    m.save(r"data/model20KRPvKRP.h5")
    #print(end_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
    #print("Training time: ",end_time - start_time)

    # calculate and print training, test loss, accuracy
    #y_train = m.predict(X2, batch_size=1024*2, verbose=1)
    #y_train_cat=np.argmax(y_train,1)-2
    #acc = np.sum(y_train_cat==df2.wdl)/y_train_cat.shape[0]
    #n_errors = y_train_cat.shape[0] - np.sum(y_train_cat==df2.wdl)
    #print("\n\nloss\nval_loss\nacc")
    #_ = [print(x) for x in hist.history["loss"]]
    #_ = [print(x) for x in hist.history["val_loss"]]
    #print(acc)
    acc_hist.append(hist.history["acc"][0])
    val_acc_hist.append(hist.history["val_acc"][0])
    loss_hist.append(hist.history["loss"][0])
    val_loss_hist.append(hist.history["val_loss"][0])

print("\n\nAll Python Iterations\nloss\nval_loss\nacc\nval_acc")
_ = [print(x) for x in loss_hist]
print()
_ = [print(x) for x in val_loss_hist]
print()
_ = [print(x) for x in acc_hist]
print()
_ = [print(x) for x in val_acc_hist]
print()
