from typing import List, Sequence

import numpy as np


def cat_if(arrays: Sequence[np.array], axis=0) -> np.array:
    '''将 array 按指定axis cat 在一起，不过忽略空的 List 或 array
    如果全部为空或列表为空，返回空 array'''

    l = [a for a in arrays if len(a)]
    if len(l):
        return np.concatenate(l, axis=axis)
    return np.array([])

def read_txt_file(file_path: str) -> List[str]:
    '''读取 txt 文件，每一行是一个路径，返回该列表
    自动忽略空的行'''

    with open(file_path, 'r') as fr:
        path_list = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]
    pl = []
    for p in path_list:
        with open(p.replace('RGB', 'txt').replace('.jpg', '.txt'), 'r') as fr:
            s = fr.read()
        if s.strip():
            pl.append(p)
    return pl
