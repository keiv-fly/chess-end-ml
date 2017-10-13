import chess
import chess.syzygy
import itertools
import re
import pandas as pd
import numpy as np
import numba
import datetime

board_start = chess.Board("K1k5/P7/8/8/8/8/8/8 w - - 0 0")
board_start0 = chess.Board("8/8/8/8/8/8/8/8 w - - 0 0")

l_boards = []
l_wdl = []
pieces = ["K", "R", "P", "k", "r", "p"]
pc_num = len(pieces)
pieces_types = [6, 4, 1, 6, 4, 1]
pieces_color = [True, True, True, False, False, False]
pieces_chess = [chess.Piece.from_symbol(s) for s in pieces]
i_min = [0,0,1,0,0,1]
i_max = [8,8,7,8,8,7]
j_min = [0,0,0,0,0,0]
j_max = [8,8,8,8,8,8]


l_boards = []
l_wdl = []
board = board_start0.copy()
i_counter = 0
n_iter=25000000
with chess.syzygy.open_tablebases(r"C:\Users\Lenovo\Downloads\syzygy") as tablebases:
    for ii in range(n_iter):
        if i_counter % 100000 == 0:
            print(i_counter, " of ", n_iter,", time: ", datetime.datetime.now().strftime("%H:%M:%S.%f"))
            print(board.fen())
        board = board_start0.copy()
        i_board = []
        i_board_i_j = []
        turn = True
        board.turn = turn
        i_board.append(turn)
        i_piece = 0
        for i_piece in range(6):
            i = np.random.randint(i_min[i_piece],i_max[i_piece])
            j = np.random.randint(j_min[i_piece],j_max[i_piece])
            board.set_piece_at(i * 8 + j, pieces_chess[i_piece])
            i_board.append(i * 8 + j)
            i_board_i_j.append(i)
            i_board_i_j.append(j)
        if len(i_board)!=len(set(i_board)):
            i_counter = i_counter + 1
            continue
        if board.is_valid():
            l_boards.append(i_board)
            l_wdl.append(tablebases.probe_wdl(board)) #*(turn*2-1))
        i_counter = i_counter + 1

df1=pd.DataFrame(l_boards)
df1.columns = ["move", "K", "R", "P", "k", "r", "p"]
df1["wdl"] = l_wdl
df2=df1.drop_duplicates()
print(len(df2.index))
print(df2.wdl.value_counts())
df2 = df2.reset_index(drop=True)

df2.to_hdf("data/KRPvKRP_table_10M_random_v2_03.h5", 'df1', complib='blosc:lz4', complevel=9)




#df2[:5000000].to_hdf("data/KRPvKRP_table_5M_random_v2.h5", 'df1', complib='blosc:lz4', complevel=9)
#df2[5000000:5010240].reset_index(drop=True).to_hdf("data/KRPvKRP_table_10K_random_test_v2.h5", 'df1', complib='blosc:lz4', complevel=9)

#df2=pd.read_hdf("data/KRPvKRP_table_10M_random_v2.h5", 'df1')
#df2[5000000:10000000].reset_index(drop=True).to_hdf("data/KRPvKRP_table_5M_random_v2_02.h5", 'df1', complib='blosc:lz4', complevel=9)
#df2[10000000:10010240].reset_index(drop=True).to_hdf("data/KRPvKRP_table_10K_random_test_v2_02.h5", 'df1', complib='blosc:lz4', complevel=9)
