from numba import njit


@njit
def test(args):
    return max(args)


print(test((True, False, None)))
