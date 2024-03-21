def f(x, y, z):
    class Point2dSimple(object):
        __match_args__ = ("x", "y")
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class Point3d(object):
        __match_args__ = ("x", "y", "z")
        def __init__(self, x=-1, y=-1, z=-1):
            self.x = x
            self.y = y
            self.z = z

    some_arbitrary_variable = x + y + z

    if z < 0:
        point = Point2dSimple(x, y)
    else:
        point = Point3d(x=x, y=y, z=z)

    match point:
        case Point2dSimple(1, 1):
            return 1 + some_arbitrary_variable
        case Point3d(1, 1, 1):
            return 3 + some_arbitrary_variable
        case Point2dSimple():
            return 4 + some_arbitrary_variable
        case Point3d(1, 1, -1):
            return 5 + some_arbitrary_variable

    return None

if __name__ == "__main__":
    print(f(1, 1, 1))
    print(f(1,1,-1))
