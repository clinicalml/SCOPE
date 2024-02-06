import torch


def y_to_seq(Y, E, shift=0):
    Y = Y-shift
    Y_int = Y.long()
    seq_y = torch.zeros(Y_int.shape[0], Y_int.max()+1, device=Y.device)
    seq_e = torch.zeros(Y_int.shape[0], Y_int.max()+1, device=Y.device)
    seq_y[(torch.arange(Y_int.shape[0]), Y_int[:, 0])] = 1
    seq_e[(torch.arange(Y_int.shape[0]), Y_int[:, 0])] = E[:, 0]

    assert (seq_y.max() == 1)
    assert (seq_e.max() == 1)
    return seq_y, seq_e
