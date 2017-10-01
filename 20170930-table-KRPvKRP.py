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
iboard0=[]
iboard0.append(bool(i_file % 2))
iboard0.append((i_file // 2) % 32)
iboard0.append((i_file // 64) % 32)

ii_start=[x[0] for x in i_lists]
board_start = board_start0.copy()
board_start.turn = bool(i_file % 2)

for i_piece in range(2):
    i = ii_start[i_piece * 2]
    j = ii_start[i_piece * 2 + 1]
    board_start.set_piece_at(i * 8 + j, pieces_chess[i_piece])

def calc_iter(i_lists, board_start, pc_num, pieces_chess, tablebases):
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
        i_board = iboard0.copy()
        for i_piece in range(2,pc_num):
            i = ii[i_piece * 2]
            j = ii[i_piece * 2 + 1]
            board.set_piece_at(i * 8 + j, pieces_chess[i_piece])
            i_board.append(i * 8 + j)
        if board.is_valid() and len(''.join(x for x in board.board_fen() if x.isalpha())) == pc_num:
            l_boards.append(i_board)
            l_wdl.append(tablebases.probe_wdl(board))
        i_counter = i_counter + 1
    return l_boards, l_wdl

#calc_iter_nb = numba.jit(calc_iter)



with chess.syzygy.open_tablebases(r"C:\Users\Lenovo\Downloads\syzygy") as tablebases:
    l_boards, l_wdl = calc_iter(i_lists, board_start, pc_num, pieces_chess,tablebases)

df1=pd.DataFrame(l_boards)
df1.columns = ["move", "K", "R", "P", "k", "r", "p"]
df1["wdl"] = l_wdl
df1.to_hdf("data/KRPvKRP_table_" + '{:02d}'.format(i_file) + ".h5", 'df1', complib='blosc:lz4', complevel=9)