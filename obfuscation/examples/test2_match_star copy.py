def f(x):

    new_x = x
    match new_x:
        case [1,2, *rest]:
            return rest
        case [*_]:
            return new_x

    return None
