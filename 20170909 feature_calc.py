import chess
import pandas as pd
import numpy as np



df = pd.read_csv("data\KPvK_1_fen.csv")
a_fen=df.values

board_shape=(8,8)

fen = a_fen[0,0]
def pieces_features(fen):
    #fen = fen [0]
    board = chess.Board(fen)
    ftrs=np.zeros((3,*board_shape))
    pos = board.king(True)
    ftrs[0,pos // 8,pos % 8] = 1
    pos = board.king(False)
    ftrs[1, pos // 8, pos % 8] = 1
    pos = list(board.pieces(chess.PAWN,True))[0]
    ftrs[2, pos // 8, pos % 8] = 1
    return ftrs

a_fen=a_fen.flatten()
l_fen=list(a_fen)
data=np.zeros((len(l_fen),3,8,8))
for i,fen in enumerate(l_fen):
    if i % 10000 == 0:
        print(i)
    data[i]=pieces_features(fen)
np.save("data\KPvK.npy",data)