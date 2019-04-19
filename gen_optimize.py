# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import datetime
import chess
import chess.syzygy

from lib import generate_table
from pathlib import Path
import datetime
import pandas
import os


# %%
# %%time
df= generate_table(100000)


# %%
n_iter = 100000
board_start = chess.Board("K1k5/P7/8/8/8/8/8/8 w - - 0 0")
board_start0 = chess.Board("8/8/8/8/8/8/8/8 w - - 0 0")

l_boards = []
l_wdl = []
pieces = ["K", "R", "P", "k", "r", "p"]
pc_num = len(pieces)
pieces_types = [6, 4, 1, 6, 4, 1]
pieces_color = [True, True, True, False, False, False]
pieces_chess = [chess.Piece.from_symbol(s) for s in pieces]
i_min = [0, 0, 1, 0, 0, 1]
i_max = [8, 8, 7, 8, 8, 7]
j_min = [0, 0, 0, 0, 0, 0]
j_max = [8, 8, 8, 8, 8, 8]

l_boards = []
l_wdl = []
board = board_start0.copy()
i_counter = 0

with chess.syzygy.open_tablebase(r"syzygy") as tablebases:
    for ii in range(n_iter):
        if i_counter % 100000 == 0:
            print(i_counter, " of ", n_iter, ", time: ", datetime.datetime.now().strftime("%H:%M:%S.%f"))
            print(board.fen())
        board = board_start0.copy()
        i_board = []
        i_board_i_j = []
        turn = True
        board.turn = turn
        i_board.append(turn)
        i_piece = 0
        for i_piece in range(6):
            i = np.random.randint(i_min[i_piece], i_max[i_piece])
            j = np.random.randint(j_min[i_piece], j_max[i_piece])
            board.set_piece_at(i * 8 + j, pieces_chess[i_piece])
            i_board.append(i * 8 + j)
            i_board_i_j.append(i)
            i_board_i_j.append(j)
        if len(i_board) != len(set(i_board)):
            i_counter = i_counter + 1
            continue
        if board.is_valid():
            l_boards.append(i_board)
            l_wdl.append(tablebases.probe_wdl(board))  # *(turn*2-1))
        i_counter = i_counter + 1

df1 = pd.DataFrame(l_boards)
df1.columns = ["move", "K", "R", "P", "k", "r", "p"]
df1["wdl"] = l_wdl
df2 = df1.drop_duplicates()
print(len(df2.index))
print(df2.wdl.value_counts())
df2 = df2.reset_index(drop=True)
 


# %%
# %load_ext line_profiler


# %%
# %lprun -f generate_table generate_table(10000)


# %%
tablebases=chess.syzygy.open_tablebase(r"syzygy")

# %%
if i_counter % 100000 == 0:
    print(i_counter, " of ", n_iter, ", time: ", datetime.datetime.now().strftime("%H:%M:%S.%f"))
    print(board.fen())
board = board_start0.copy()
i_board = []
i_board_i_j = []
turn = True
board.turn = turn
i_board.append(turn)
i_piece = 0
for i_piece in range(6):
    i = np.random.randint(i_min[i_piece], i_max[i_piece])
    j = np.random.randint(j_min[i_piece], j_max[i_piece])
    board.set_piece_at(i * 8 + j, pieces_chess[i_piece])
    i_board.append(i * 8 + j)
    i_board_i_j.append(i)
    i_board_i_j.append(j)
board.is_valid()

# %%
# %lprun -f tablebases.probe_wdl tablebases.probe_wdl(board)

# %%
# %lprun -f tablebases.probe_ab tablebases.probe_ab(board,-2,2)

# %%
# %lprun -f tablebases.probe_wdl_table tablebases.probe_wdl_table(board)

# %%
key = chess.syzygy.calc_key(board)

# %%
table = tablebases.wdl[key]

# %%
tablebases._bump_lru(table)

# %%
# %lprun -f table.probe_wdl_table table.probe_wdl_table(board)

# %%
# %lprun -f table._probe_wdl_table table._probe_wdl_table(board)

# %%
# %lprun -f chess.syzygy.calc_key chess.syzygy.calc_key(board)

# %%
key

# %%
# %%time
t=board.occupied_co[not board.turn]

# %%
# %%time
board.generate_legal_moves(to_mask=t)

# %%
# %%time
run(t)


# %%
def run(t):
    list(board.generate_legal_moves(to_mask=t))


# %%
# %lprun -f board.generate_legal_moves board.generate_legal_moves(to_mask=t)

# %%
# %lprun -f board.generate_legal_moves run(t)

# %%
board.generate_legal_moves(to_mask=t)

# %%
# %lprun -f board.generate_pseudo_legal_moves list(board.generate_pseudo_legal_moves(chess.BB_ALL, t))

# %%
from_mask=chess.BB_ALL
to_mask=t
our_pieces = board.occupied_co[board.turn]
non_pawns = our_pieces & ~board.pawns & from_mask
t1=list(chess.scan_reversed(non_pawns))[0]

# %%
# %lprun -f board.attacks_mask board.attacks_mask(t1)

# %%
t1

# %%
# %%time
board.attacks_mask(t1)

# %%
SQUARES = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
] = range(64)

# %%
SQUARES

# %%
BB_SQUARES = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
    BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
    BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8
] = [1 << sq for sq in SQUARES]

# %%
BB_SQUARES

# %%
