a = [0, 1, 2, 1, 2, 0, 3, 5, 3, 2]
c = [0, 1, 9, 1, 9, 0, 9, 5, 3, 2]

b = [i for i in a if i != 2]
d = [c[i] for i in range(len(a)) if a[i] != 2]
print(b)
print(d)
