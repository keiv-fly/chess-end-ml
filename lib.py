import numpy as np
import pandas as pd
import datetime
import chess
import chess.syzygy

def features_from_table(df):
    n_rows = len(df.index)

    data = np.zeros((n_rows, 16, 8, 8), dtype=np.float16)
    ii = 0
    #print(datetime.datetime.now().strftime("%H:%M:%S.%f"))
    #print(ii)
    ii = ii + 1
    for i in range(int(np.ceil(n_rows / 1000))):
        m_i = min(1000 * (i + 1), n_rows)
        idx = list(range(i * 1000, m_i))
        data[idx, 0, df.K[idx].values // 8, df.K[idx] % 8] = 1
    #print(ii)
    ii = ii + 1
    for i in range(int(np.ceil(n_rows / 1000))):
        m_i = min(1000 * (i + 1), n_rows)
        idx = list(range(i * 1000, m_i))
        data[idx, 1, df.k[idx] // 8, df.k[idx] % 8] = 1
    #print(ii)
    ii = ii + 1
    for i in range(int(np.ceil(n_rows / 1000))):
        m_i = min(1000 * (i + 1), n_rows)
        idx = list(range(i * 1000, m_i))
        data[idx, 2, df.R[idx] // 8, df.R[idx] % 8] = 1
    #print(ii)
    ii = ii + 1
    for i in range(int(np.ceil(n_rows / 1000))):
        m_i = min(1000 * (i + 1), n_rows)
        idx = list(range(i * 1000, m_i))
        data[idx, 3, df.r[idx] // 8, df.r[idx] % 8] = 1
    #print(ii)
    ii = ii + 1
    for i in range(int(np.ceil(n_rows / 1000))):
        m_i = min(1000 * (i + 1), n_rows)
        idx = list(range(i * 1000, m_i))
        data[idx, 4, df.P[idx] // 8, df.P[idx] % 8] = 1
    #print(ii)
    ii = ii + 1
    for i in range(int(np.ceil(n_rows / 1000))):
        m_i = min(1000 * (i + 1), n_rows)
        idx = list(range(i * 1000, m_i))
        data[idx, 5, df.p[idx] // 8, df.p[idx] % 8] = 1
    #print(ii)
    ii = ii + 1
    data[:, 6, :, :] = ((df.K // 8) / 7)[:, np.newaxis, np.newaxis]
    #print(ii)
    ii = ii + 1
    data[:, 7, :, :] = ((df.k // 8) / 7)[:, np.newaxis, np.newaxis]
    #print(ii)
    ii = ii + 1
    data[:, 8, :, :] = ((df.R // 8) / 7)[:, np.newaxis, np.newaxis]
    #print(ii)
    ii = ii + 1
    data[:, 9, :, :] = ((df.r // 8) / 7)[:, np.newaxis, np.newaxis]
    #print(ii)
    ii = ii + 1
    data[:, 10, :, :] = ((df.P // 8) / 7)[:, np.newaxis, np.newaxis]
    #print(ii)
    ii = ii + 1
    data[:, 11, :, :] = ((df.p // 8) / 7)[:, np.newaxis, np.newaxis]
    #print(ii)
    ii = ii + 1
    data[:, 12, :, :] = (np.maximum(abs((df.P // 8) - (df.K // 8)), abs((df.P % 8) - (df.K % 8))) / 8)[:, np.newaxis,
                        np.newaxis]
    #print(ii)
    ii = ii + 1
    data[:, 13, :, :] = (np.maximum(abs((df.P // 8) - (df.k // 8)), abs((df.P % 8) - (df.k % 8))) / 8)[:, np.newaxis,
                        np.newaxis]
    #print(ii)
    ii = ii + 1
    data[:, 14, :, :] = (np.maximum(abs((df.p // 8) - (df.K // 8)), abs((df.P % 8) - (df.K % 8))) / 8)[:, np.newaxis,
                        np.newaxis]
    #print(ii)
    ii = ii + 1
    data[:, 15, :, :] = (np.maximum(abs((df.p // 8) - (df.k // 8)), abs((df.P % 8) - (df.k % 8))) / 8)[:, np.newaxis,
                        np.newaxis]
    #print(ii)
    ii = ii + 1
    #print(datetime.datetime.now().strftime("%H:%M:%S.%f"))
    return data

def generate_table(n_iter = 25000000):
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

    with chess.syzygy.open_tablebases(r"syzygy") as tablebases:
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
    return df2
