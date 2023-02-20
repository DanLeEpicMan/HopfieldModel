'''
A module containing code for a Hopfield Model neural network. 
Patterns are `n`-sized arrays whose entires (nodes) are either 1 or -1.
    

By default, weights are computed via superposition, i.e. `w_{ij}` is the sum of `p_{i} * p_{j}` for each pattern
p, and dividing everything by the total number of patterns.\n
New nodes are computed via linear combinations of the weights and applying the `sgn` function.\n
As a result of the above two, both patterns and their analogs will be stable attractors. Certain linear
combinations of an odd number of patterns (mixture states) also act as attractors.

Everything explained above happens by default; almost everything can be configured. See docstrings for `Network`.
'''
import math
import numpy as np
from typing import Callable


class Network:
    '''
    A class modeling a Hopfield neural network. 
    Patterns and state configurations are understood to be sequences of 1's and -1's.

    If you would like to configure this class, then subclass and override methods.

    # Attributes
    The following attributes are read-only.
      `patterns`: The given initial patterns. 
      All patterns are stable attractors, but not all stable attractors are patterns (see `compute` docstring)
      `weights`: The computed weights from `patterns`. See `__init__` docstring for how they're computed.
      `P`: The total number of patterns.
      `N`: The total number of information (nodes) in each pattern.

    # Methods
      `sgn`: Returns the sign of x; 1 if x is positive or 0, -1 if x is negative.
      `compute`: Transforms the given state configuration into a stable attractor.
    '''
    def __init__(self, patterns: np.ndarray, *, omit_symmetric_weights: bool = True, compute_weights: Callable[[np.ndarray, int, int], np.ndarray] = None) -> None:
        '''
        Initializes the Hopfield model by creating a weight matrix.
        ## Parameters
        `patterns`: An `m x n` matrix. Each row represents a pattern, each with `n` pieces of information (1 or -1).
        
        E.g. The following array is 3 patterns with 4 bits of information each
        ```
        [[1, 1, 1, 1],
        [-1, 1, -1, 1],
        [1, 1, -1, -1]]
        ```
        `omit_symmetric_weights = True`: If true, then symmetric weights (`w_{ii}`) will be set to 0 instead of computed.\n
        `compute_weights = None`: If given, this function will be used to compute weights instead of the default superposition of terms.\n
        (I.e. By default `w_{ij}` is the sum of `p_{i} * p_{j}` for each pattern `p`, and dividing everything by the total number of patterns.)\n
        This will be called like so:
        ```python
            self.weights = compute_weights(patterns, N, P)
        ``` 
        And is expected to return an `N x N` matrix of weights, where `w_{ij}` is how the `i`th node is affected by the `j`th node.
        '''
        self._patterns = patterns
        self._P = len(patterns)
        self._N = len(patterns[0])

        if compute_weights is not None:
            self._weights = compute_weights(patterns, self._N, self._P)
        else:
            self._weights = np.empty((self._N, self._N), dtype=np.float64)
            # since the weights are symmetric (w_{ij} = w_{ji}), 
            # it is only necessary to loop over the lower triangle of the weight matrix.
            for i in range(self._N):
                for j in range(i + 1):
                    if i==j and omit_symmetric_weights:
                        self._weights[i, i] = 0
                        continue
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
        Computes the stable configuration of the given state based on `weights`.

        By default, every pattern is a stable attractor, but they are not the only ones. 
        Negative analogs are also stable attractors, along with states that are equidistant to other attractors.
        Sometimes there may exist other attractors.

        ### Raises 
        `ValueError`: The given state is not the same size as the patterns.
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
