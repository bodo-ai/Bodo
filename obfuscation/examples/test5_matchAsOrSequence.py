def f(x, y, z):

    def w():
        return -5
    some_arbitrary_variable = x + y + z

    new_x = x

    match new_x:
        case [1,2,3] | -1 as new_var:
            return new_var + some_arbitrary_variable
        case 4 | -4 as y:
            return y * 4 + y + some_arbitrary_variable
        case 5 as z:
            return z + y + some_arbitrary_variable
        case _ if w() > z:
            return z + some_arbitrary_variable
        case new_x:
            return new_x + some_arbitrary_variable

    return None

if __name__ == "__main__":
    print(f(1, 1, 1))
    print(f(2,1,-1))
    print(f(5,10,-11))
    print(f(4,1234,108))
    print(f(-1, -1, -1))
