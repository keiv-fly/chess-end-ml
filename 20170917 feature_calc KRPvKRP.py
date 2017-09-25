import chess
import pandas as pd
import numpy as np

i_file=2
df=pd.read_hdf("data/KRPvKRP_fen_" + '{:02d}'.format(i_file) + ".h5", 'df1')
a_fen=df[0].values

board_shape=(8,8)

l_fen=list(a_fen)

fen = l_fen[0]
def pieces_features(fen):
    #fen = fen [0]
    board = chess.Board(fen)
    ftrs=np.zeros((19,*board_shape))
    ftrs[0, :, :] = board.turn
    pos1 = board.king(True)
    ftrs[1,pos1 // 8,pos1 % 8] = 1
    pos2 = board.king(False)
    ftrs[2, pos2 // 8, pos2 % 8] = 1
    pos3 = list(board.pieces(chess.ROOK,True))[0]
    ftrs[3, pos3 // 8, pos3 % 8] = 1
    ftrs[4, :, :] = (pos3 // 8) / 7
    pos4 = list(board.pieces(chess.ROOK,False))[0]
    ftrs[5, pos4 // 8, pos4 % 8] = 1
    ftrs[6, :, :] = (pos4 // 8) / 7

    pos5 = list(board.pieces(chess.PAWN,True))[0]
    ftrs[7, pos5 // 8, pos5 % 8] = 1
    ftrs[8, :, :] = np.maximum(abs((pos5 // 8) - (pos1 // 8)), abs((pos5 % 8) - (pos1 % 8)))/8
    ftrs[9, :, :] = np.maximum(abs((pos5 // 8) - (pos2 // 8)), abs((pos5 % 8) - (pos2 % 8)))/8
    ftrs[10, :, :] = ((pos1 // 8) - 1) / 7
    ftrs[11, :, :] = ((pos2 // 8) - 1) / 7
    ftrs[12, :, :] = ((pos5 // 8) - 1) / 7
    ftrs[13, :, :] = (pos5 % 8) / 7

    pos6 = list(board.pieces(chess.PAWN,False))[0]
    ftrs[14, pos6 // 8, pos6 % 8] = 1
    ftrs[15, :, :] = np.maximum(abs((pos6 // 8) - (pos1 // 8)), abs((pos6 % 8) - (pos1 % 8)))/8
    ftrs[16, :, :] = np.maximum(abs((pos6 // 8) - (pos2 // 8)), abs((pos6 % 8) - (pos2 % 8)))/8
    ftrs[17, :, :] = ((pos6 // 8) - 1) / 7
    ftrs[18, :, :] = (pos6 % 8) / 7
    return ftrs


data=np.zeros((len(l_fen),19,8,8))
for i,fen in enumerate(l_fen):
    if i % 10000 == 0:
        print(i)
    data[i]=pieces_features(fen)
np.save("data\KRPvKRP_19f.npy",data)