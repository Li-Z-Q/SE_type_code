import random

a = [0, 1, 2, 1, 2, 0, 3, 5, 3, 2]
c = [0, 1, 9, 1, 9, 0, 9, 5, 3, 2]

random.seed(0)
random.shuffle(a)
print(a)