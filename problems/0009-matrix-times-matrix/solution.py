def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:
    import numpy as np
    a_array = np.array(a)
    b_array = np.array(b)
    if (a_array.shape[1] == b_array.shape[0]):
        c = a_array @ b_array
    else:
        c = -1
    return c