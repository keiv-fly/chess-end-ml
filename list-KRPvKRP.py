import chess
import chess.syzygy
import itertools
import re
import pandas as pd
import numpy as np
import numba
import datetime

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
i_max = [7,7,6,7,7,6]


i_lists = []
for i in range(pc_num):
    i_lists.append(list(range(i_min[i],i_max[i]+1)))
    i_lists.append(list(range(8)))

i_file=2
i_lists[0]=[((i_file // 2) % 32) // 4]
i_lists[1]=[(i_file // 2) % 4]
i_lists[2]=[((i_file // 64) % 32) // 8]
i_lists[3]=[(i_file // 64) % 8]

ii_start=[x[0] for x in i_lists]
board_start = board_start0.copy()
board_start.turn = bool(i_file % 2)
for i_piece in range(2):
    i = ii_start[i_piece * 2]
    j = ii_start[i_piece * 2 + 1]
    board_start.set_piece_at(i * 8 + j, pieces_chess[i_piece])

def calc_iter(i_lists, board_start, pc_num, pieces_chess):
    board = board_start
    l_boards = []
    l_wdl = []
    n_iter = np.prod(np.array([len(x) for x in i_lists]))
    i_counter = 0
    for ii in itertools.product(*i_lists):
        if i_counter % 100000 == 0:
            print(i_counter, " of ", n_iter,", time: ", datetime.datetime.now().strftime("%H:%M:%S.%f"))
            print(board.fen())
        board = board_start.copy()
        for i_piece in range(2,pc_num):
            i = ii[i_piece * 2]
            j = ii[i_piece * 2 + 1]
            board.set_piece_at(i * 8 + j, pieces_chess[i_piece])
        if board.is_valid() and len(''.join(x for x in board.board_fen() if x.isalpha())) == pc_num:
            l_boards.append(board.fen())
            #l_wdl.append(tablebases.probe_wdl(board))
        i_counter = i_counter + 1
    return l_boards

#calc_iter_nb = numba.jit(calc_iter)
l_boards = calc_iter(i_lists, board_start, pc_num, pieces_chess)

def calc_iter_wdl(i_lists, board_start, pc_num, pieces_chess, tablebases):
    l_boards = []
    l_wdl = []
    n_iter = np.prod(np.array([len(x) for x in i_lists]))
    i_counter = 0
    for ii in itertools.product(*i_lists):
        if i_counter % 100000 == 0:
            print(i_counter, " of ", n_iter,", time: ", datetime.datetime.now().strftime("%H:%M:%S.%f"))
        board = board_start.copy()
        for i_piece in range(2,pc_num):
            i = ii[i_piece * 2]
            j = ii[i_piece * 2 + 1]
            board.set_piece_at(i * 8 + j, pieces_chess[i_piece])
        if board.is_valid() and len(''.join(x for x in board.board_fen() if x.isalpha())) == pc_num:
            #l_boards.append(board.fen())
            l_wdl.append(tablebases.probe_wdl(board))
        i_counter = i_counter + 1
    return l_wdl

with chess.syzygy.open_tablebases(r"C:\Users\Lenovo\Downloads\syzygy") as tablebases:
    l_wdl = calc_iter_wdl(i_lists, board_start, pc_num, pieces_chess, tablebases)

df1=pd.DataFrame(l_boards)
df1.to_hdf("data/KRPvKRP_fen_" + '{:02d}'.format(i_file) + ".h5", 'df1', complib='blosc:lz4', complevel=9)
df1=pd.read_hdf("data/KRPvKRP_fen_" + '{:02d}'.format(i_file) + ".h5", 'df1')
df2=pd.DataFrame(l_wdl)
df2.to_hdf("data/KRPvKRP_wdl_" + '{:02d}'.format(i_file) + ".h5", 'df2', complib='blosc:lz4', complevel=9)

