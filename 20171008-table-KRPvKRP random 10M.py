import chess
import chess.syzygy
import itertools
import re
import pandas as pd
import numpy as np
import numba
import datetime

def valid_board(turn,i_board_i_j,i_board):
    if turn:
        if i_board_i_j[2] == i_board_i_j[6]:
            return False
        if i_board_i_j[3] == i_board_i_j[7]:
            return False
        if i_board[3] == i_board[4]-7:
            return False
        if i_board[3] == i_board[4]-9:
            return False
    else:
        if i_board_i_j[0] == i_board_i_j[8]:
            return False
        if i_board_i_j[1] == i_board_i_j[9]:
            return False
        if i_board[6] == i_board[1]+7:
            return False
        if i_board[6] == i_board[1]+9:
            return False
    if max(abs(i_board_i_j[0]-i_board_i_j[6]),abs(i_board_i_j[1]-i_board_i_j[7])) == 1:
        return False
    return True

regex = re.compile(r'[a-zA-Z]')

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
n_iter=100001
with chess.syzygy.open_tablebases(r"C:\Users\Lenovo\Downloads\syzygy") as tablebases:
    for ii in range(n_iter):
        if i_counter % 100000 == 0:
            print(i_counter, " of ", n_iter,", time: ", datetime.datetime.now().strftime("%H:%M:%S.%f"))
            print(board.fen())
        board = board_start0.copy()
        i_board = []
        i_board_i_j = []
        turn = np.random.uniform()<0.5
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
        #print(board)
        # if len(i_board)!=len(set(i_board)):
        #     i_counter = i_counter + 1
        #     continue
        if board.is_valid() and len(''.join(x for x in board.board_fen() if x.isalpha())) == pc_num:
        #if board.is_valid():
        #if valid_board(turn,i_board_i_j,i_board):
            l_boards.append(i_board)
            l_wdl.append(tablebases.probe_wdl(board)*(turn*2-1))
        i_counter = i_counter + 1

df1=pd.DataFrame(l_boards)
df1.columns = ["move", "K", "R", "P", "k", "r", "p"]
df1["wdl"] = l_wdl
df2=df1.drop_duplicates()
df2.wdl.value_counts()

df2.to_hdf("data/KRPvKRP_table_10M_random.h5", 'df1', complib='blosc:lz4', complevel=9)
df2.reset_index(drop=True)[:5000000].to_hdf("data/KRPvKRP_table_5M_random.h5", 'df1', complib='blosc:lz4', complevel=9)