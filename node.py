import math
import numpy as np


class Network:
    '''
    A Hopfield Model neural network. 
    Patterns are `n`-sized arrays whose entires (nodes) are either 1 or -1.
    
    Weights are computed via superposition, i.e. `w_{ij}` is the sum of `p_{i} * p_{j}` for each pattern
    p, and dividing everything by the total number of patterns.\n
    New nodes are computed via linear combinations of the weights and applying the `sgn` function.\n
    As a result of the above two, both patterns and their analogs will be stable attractors.
    '''
    def __init__(self, patterns: np.ndarray) -> None:
        '''
        Creates a Hopfield-model neural network with superpositioned weights.
        ## Parameters
        `patterns`: An `m x n` matrix. Each row represents a pattern, each with `n` pieces of information (1 or -1).
        
        E.g. The following array is 3 patterns with 4 bits of information each
        ```
        [[1, 1, 1, 1],
        [-1, 1, -1, 1],
        [1, 1, -1, -1]]
        ```
        ## Attributes
        The following attributes are read-only.
          `patterns`: The given initial patterns.
          `weights`: The computed weights from `patterns`. Based on superposition of terms.
          `P`: The total number of patterns.
          `N`: The total number of information in each pattern.
        '''
        self._patterns = patterns
        self._P = len(patterns)
        self._N = len(patterns[0])

        self._weights = np.empty((self._N, self._N), dtype=np.float64)
        # since the weights are symmetric (w_{ij} = w_{ji}), 
        # it is only necessary to loop over the lower triangle of the weight matrix.
        for i in range(self._N):
            for j in range(i + 1):
                weight_sum = 0.0
                for k in range(self._P):
                    weight_sum += patterns[k, i] * patterns[k, j]
                self._weights[i, j] = self._weights[j, i] = 1/self._N * weight_sum 

    @property
    def P(self):
        '''
        The total number of patterns.
        '''
        return self._P

    @property
    def N(self):
        '''
        The amount of information each pattern has.
        '''
        return self._N

    @property
    def weights(self):
        '''
        The weights computed from the given set of patterns.

        Note that the weights are a superposition of terms; that is, "averaged" among all weights.
        '''
        return self._weights

    @property
    def patterns(self):
        '''
        The set of initial patterns.
        '''
        return self._patterns

    @staticmethod
    def sgn(x: float, /) -> int:
        '''
        Returns the sign of x. 0 is treated as positive
        '''
        return 1 if x >=0 else -1

    def compute(self, initial_state: np.ndarray, /) -> np.ndarray:
        '''
        Computes the stable configuration of the given state.

        A ValueError is raised if the given state is not the same size as the patterns.
        '''
        if len(initial_state) != self.N: 
            raise ValueError(f"Invalid input size given. Expected {self.N}, got {len(initial_state)}")
        previous = np.zeros(1)
        current = initial_state
        while not np.array_equal(previous, current):
            previous = current
            for i in range(len(current)):
                state_sum = 0.0
                for j in range(len(current)):
                    state_sum += self.weights[i, j] * current[j]
                current[i] = self.sgn(state_sum)
        return current



net = Network(patterns=np.array([
    [1, 1, 1, -1, 1, 1],
    [-1, 1, -1, -1, 1, -1]
], dtype=np.int8))

print(net.compute([-1, 1, 1, 1, -1, 1]))
