import numpy as np

def average(l):
    llen = len(l)

    def divide(x): return x / llen

    return map(divide, map(sum, zip(*l)))

A = np.array([[0.875, 0.875, 0.1875, 0., 0.],
              [1., 1., 0.5625, 0.375, 0.125, ],
              [0.75, 1., 0.5625, 0., 0.25],
              [0.375, 0.5, 0.6875, 0.5, 0.875],
              [1., 0.625, 0.6875, 0.5, 0.375],
              [1., 1., 0.5, 0.375, 0.5],
              [1., 0.875, 0.53333333, 0., 0.125]])

print(A[1])

needed_lists = []
for i in range(len(A)):
    inner_list = A[i]
    inner_element = []
    for element in [0,1,3,4]:
        elements = inner_list[element]
        inner_element.append(elements)

        needed_lists.append(inner_element)
