import pandas as pd
import numpy as np
import datetime
import numba

i_file=2
df=pd.read_hdf("data/KRPvKRP_table_" + '{:02d}'.format(i_file) + ".h5", 'df1')

#df=df.head(10000)



def run(df):
    board_shape = (8, 8)

    n_rows = len(df.index)

    data = np.zeros((n_rows, 19, 8, 8), dtype=np.float16)
    i_board = df.iloc[0, :]
    # def pieces_features(i_board):
    #     ftrs = np.zeros((19, *board_shape))
    #     ftrs[0, :, :] = i_board.move
    #     pos1 = i_board.K
    #     ftrs[1, pos1 // 8, pos1 % 8] = 1
    #     pos2 = i_board.k
    #     ftrs[2, pos2 // 8, pos2 % 8] = 1
    #     pos3 = i_board.R
    #     ftrs[3, pos3 // 8, pos3 % 8] = 1
    #     ftrs[4, :, :] = (pos3 // 8) / 7
    #     pos4 = i_board.r
    #     ftrs[5, pos4 // 8, pos4 % 8] = 1
    #     ftrs[6, :, :] = (pos4 // 8) / 7
    #
    #     pos5 = i_board.P
    #     ftrs[7, pos5 // 8, pos5 % 8] = 1
    #     ftrs[8, :, :] = np.maximum(abs((pos5 // 8) - (pos1 // 8)), abs((pos5 % 8) - (pos1 % 8))) / 8
    #     ftrs[9, :, :] = np.maximum(abs((pos5 // 8) - (pos2 // 8)), abs((pos5 % 8) - (pos2 % 8))) / 8
    #     ftrs[10, :, :] = ((pos1 // 8) - 1) / 7
    #     ftrs[11, :, :] = ((pos2 // 8) - 1) / 7
    #     ftrs[12, :, :] = ((pos5 // 8) - 1) / 7
    #     ftrs[13, :, :] = (pos5 % 8) / 7
    #
    #     pos6 = i_board.p
    #     ftrs[14, pos6 // 8, pos6 % 8] = 1
    #     ftrs[15, :, :] = np.maximum(abs((pos6 // 8) - (pos1 // 8)), abs((pos6 % 8) - (pos1 % 8))) / 8
    #     ftrs[16, :, :] = np.maximum(abs((pos6 // 8) - (pos2 // 8)), abs((pos6 % 8) - (pos2 % 8))) / 8
    #     ftrs[17, :, :] = ((pos6 // 8) - 1) / 7
    #     ftrs[18, :, :] = (pos6 % 8) / 7
    #     return ftrs
    for i in range(n_rows):
        if i % 10000 == 0:
            print(i, " of ", n_rows, ", time: ", datetime.datetime.now().strftime("%H:%M:%S.%f"))
        i_board = df.iloc[i, :]
        ftrs = np.zeros((19, 8, 8))
        ftrs[0, :, :] = i_board.move
        pos1 = i_board.K
        ftrs[1, pos1 // 8, pos1 % 8] = 1
        pos2 = i_board.k
        ftrs[2, pos2 // 8, pos2 % 8] = 1
        pos3 = i_board.R
        ftrs[3, pos3 // 8, pos3 % 8] = 1
        ftrs[4, :, :] = (pos3 // 8) / 7
        pos4 = i_board.r
        ftrs[5, pos4 // 8, pos4 % 8] = 1
        ftrs[6, :, :] = (pos4 // 8) / 7

        pos5 = i_board.P
        ftrs[7, pos5 // 8, pos5 % 8] = 1
        ftrs[8, :, :] = np.maximum(abs((pos5 // 8) - (pos1 // 8)), abs((pos5 % 8) - (pos1 % 8))) / 8
        ftrs[9, :, :] = np.maximum(abs((pos5 // 8) - (pos2 // 8)), abs((pos5 % 8) - (pos2 % 8))) / 8
        ftrs[10, :, :] = ((pos1 // 8) - 1) / 7
        ftrs[11, :, :] = ((pos2 // 8) - 1) / 7
        ftrs[12, :, :] = ((pos5 // 8) - 1) / 7
        ftrs[13, :, :] = (pos5 % 8) / 7

        pos6 = i_board.p
        ftrs[14, pos6 // 8, pos6 % 8] = 1
        ftrs[15, :, :] = np.maximum(abs((pos6 // 8) - (pos1 // 8)), abs((pos6 % 8) - (pos1 % 8))) / 8
        ftrs[16, :, :] = np.maximum(abs((pos6 // 8) - (pos2 // 8)), abs((pos6 % 8) - (pos2 % 8))) / 8
        ftrs[17, :, :] = ((pos6 // 8) - 1) / 7
        ftrs[18, :, :] = (pos6 % 8) / 7
        data[i]=ftrs
    return data

run_nb = numba.jit(run)
data = run_nb(df)
np.save("data\KRPvKRP_19f.npy",data)