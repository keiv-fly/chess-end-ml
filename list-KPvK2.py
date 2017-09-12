import chess
import chess.syzygy
import itertools
import re
import pandas as pd

regex = re.compile(r'[a-zA-Z]')

board_start = chess.Board("K1k5/P7/8/8/8/8/8/8 w - - 0 0")
board_start = chess.Board("8/8/8/8/8/8/8/8 w - - 0 0")

l_boards = []
l_wdl = []
pieces = ["K", "k", "P"]
pc_num = len(pieces)
pieces_types = [6, 6, 1]
pieces_color = [True, False, True]
pieces_chess = [chess.Piece.from_symbol(s) for s in pieces]
i_min = [0,0,1]
i_max = [7,7,6]


i_lists = []
for i in range(pc_num):
    i_lists.append(list(range(i_min[i],i_max[i]+1)))
    i_lists.append(list(range(8)))


with chess.syzygy.open_tablebases(r"C:\Users\Lenovo\Downloads\syzygy") as tablebases:
    #print(tablebases.probe_wdl(chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")))
    i_counter=0
    for ii in itertools.product(*i_lists):
        if i_counter % 10000 == 0:
            print(i_counter)
        board=board_start.copy()
        for i_piece in range(pc_num):
            i=ii[i_piece*2]
            j=ii[i_piece*2 + 1]
            board.set_piece_at(i*8+j,pieces_chess[i_piece])
        if board.is_valid() and len(''.join(x for x in board.board_fen() if x.isalpha()))==pc_num:
            l_boards.append(board.fen())
            l_wdl.append(tablebases.probe_wdl(board))
        i_counter = i_counter + 1

df = pd.DataFrame({"FEN":l_boards,"wdl": l_wdl})
df.to_csv("data\KPvK_1_fen_wdl.csv", index= False)
a_fen=df.values
