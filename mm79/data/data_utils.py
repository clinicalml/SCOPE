import torch


def convert_pids_to_str(pids):
    converted_pids = []
    for pid in pids:
        pid_str = str(int(pid))
        if len(pid_str) == 7:
            pid_str = pid_str[:3] + "-" + pid_str[3:]
            converted_pids.append(pid_str)
        else:
            if len(pid_str) == 7:
                pid_str = "0"+pid_str
            converted_pids.append("-".join([pid_str[:5], pid_str[5:]]))
    return converted_pids


def pids_to_numeric(pids):
    if pids[0][0] == "C":
        pids_num = torch.Tensor(
            [int(pid[6:].replace("-", "")) for pid in pids])
    else:
        pids_num = torch.LongTensor([int("".join(p.split("-"))) for p in pids])
    return pids_num
