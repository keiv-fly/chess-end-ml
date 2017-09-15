import chess
import pandas as pd
import numpy as np



df = pd.read_csv("data\KPvK_1_fen.csv")
a_fen=df.values

board_shape=(8,8)

a_fen=a_fen.flatten()
l_fen=list(a_fen)

fen = a_fen[0]
def pieces_features(fen):
    #fen = fen [0]
    board = chess.Board(fen)
    ftrs=np.zeros((8,*board_shape))
    pos1 = board.king(True)
    ftrs[0,pos1 // 8,pos1 % 8] = 1
    pos2 = board.king(False)
    ftrs[1, pos2 // 8, pos2 % 8] = 1
    pos3 = list(board.pieces(chess.PAWN,True))[0]
    ftrs[2, pos3 // 8, pos3 % 8] = 1
    ftrs[3, :, :] = np.maximum(abs((pos3 // 8) - (pos1 // 8)), abs((pos3 % 8) - (pos1 % 8)))/8
    ftrs[4, :, :] = np.maximum(abs((pos3 // 8) - (pos2 // 8)), abs((pos3 % 8) - (pos2 % 8)))/8
    ftrs[5, :, :] = ((pos1 // 8) - 1) / 7
    ftrs[6, :, :] = ((pos2 // 8) - 1) / 7
    ftrs[7, :, :] = ((pos3 // 8) - 1) / 7
    return ftrs


data=np.zeros((len(l_fen),8,8,8))
for i,fen in enumerate(l_fen):
    if i % 10000 == 0:
        print(i)
    data[i]=pieces_features(fen)
np.save("data\KPvK_w_5.npy",data)