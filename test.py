import numpy as np
from network import Network


S = list([x1, y1, x2, y2] for x1 in range(0, 2) for x2 in range(0, 2) for y1 in range(0, 2) for y2 in range(0, 2))


def create_space(N: int):
    '''
    Creates `{-1, 1}^N`
    '''
    space = []
    def convert(l: list):
        for i in range(len(l)):
            l[i] = -1 if l[i]==0 else 1
        return l

    def looper(comb):
        '''
        comb: combinations generated thus far
        '''
        for x in range(2):
            if len(comb) == N-1:
                space.append(convert(comb + [x]))
            else:
                looper(comb + [x])
    
    looper([])
    
    return space

print(create_space(6))

net1 = Network(patterns=np.array([
    [1, 1, 1, -1, 1, 1],
    [-1, 1, -1, -1, 1, -1]
], dtype=np.int8))

net2 = Network(patterns=np.array([
    [1, 1, 1, -1, 1, 1],
    [-1, 1, -1, -1, 1, -1]
], dtype=np.int8), omit_symmetric_weights=False)

print(net1.weights)