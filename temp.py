import copy


class Text:
    def __init__(self):
        self.a = [1, 2, 3]
        self.b= 4
    def get_self(self):
        return copy.deepcopy(self)

x = Text()
print("x.a: ", x.a)
print("x.b: ", x.b)

y = x.get_self()
print("y.a: ", y.a)
print("y.b: ", y.b)

x.a[0] = 100
x.b = 100000
print("x.a: ", x.a)
print("x.b: ", x.b)


print("y.a: ", y.a)
print("y.b: ", y.b)

