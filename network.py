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
from random import random
import numpy as np
from typing import Callable


def sgn(x: float, /) -> int:
    '''
    Returns the sign of x. 0 is treated as positive
    '''
    return 1 if x >=0 else -1

class Network:
    '''
    A class modeling a Hopfield neural network. 
    Patterns and state configurations are understood to be sequences of 1's and -1's.

    If you would like to configure this class, then subclass and override methods.

    # Attributes
    The following attributes are read-only.
      `patterns`: The given initial patterns. Should be a list of -1s and 1s. `ndarrays` will automatically be reshaped.
      All patterns are stable attractors, but not all stable attractors are patterns (see `compute` docstring).
      `weights`: The computed weights from `patterns`. See `__init__` docstring for how they're computed.
      `P`: The total number of patterns.
      `N`: The total number of information (nodes) in each pattern.

    # Methods
      `compute`: Transforms the given state configuration into a stable attractor.
      `is_learned_patterns`: Returns 1 if the given pattern is an initial pattern, -1 if it's the negative of some initial pattern, and 0 otherwise.
    '''
    def __init__(self, 
            patterns: list[np.ndarray | list[int]], 
            *, 
            certainty: float = math.inf,
            omit_symmetric_weights: bool = True, 
            compute_weights: Callable[[np.ndarray, int, int], np.ndarray] = None
        ) -> None:
        '''
        Initializes the Hopfield model by creating a weight matrix and a sigmoid function.
        ## Parameters
        `patterns`: A list of patterns to be remembered, whose entires are either -1 or 1.
        The inputs can be either Python lists or `ndarrays` of any shape; 
        this will automatically reshape in the latter case.
        
        E.g. The following array is 3 patterns with 4 bits of information each
        ```
        [[1, 1, 1, 1],
        [-1, 1, -1, 1],
        [1, 1, -1, -1]]
        ```

        `certainty = math.inf`: A nonnegative number (including `+infty`) parameterizing 
        the degree of randomness used in computing new states. 
        See the `compute` docstring for an explanation on how this is used.
        
        A value of 0 corresponds to total randomness (i.e. 50% chance of being 1 and -1 regardless of the input),
        and `math.inf` corresponds to total determinism (i.e. 100% of being 1 if input is positive).\n

        `omit_symmetric_weights = True`: If true, then symmetric weights (`w_{ii}`) will be set to 0.\n

        `compute_weights = None`: If given, this function will be used to compute weights instead of the default superposition of terms.\n
        (I.e. By default `w_{ij}` is the sum of `p_{i} * p_{j}` for each pattern `p`, and dividing everything by the total number of patterns.)\n
        This will be called like so:
        ```python
            self.weights = compute_weights(patterns, N, P)
        ``` 
        And is expected to return an `N x N` matrix of weights, where `w_{ij}` is how the `i`th node is affected by the `j`th node.

        ## Raises
        `ValueError`: `certainty` is negative.
        '''
        if certainty < 0: raise ValueError("Certainty parameter must be nonnegative.")

        self._P = len(patterns)
        self._N = len(patterns[0]) if isinstance(patterns[0], list) else patterns[0].size # because the first entry can be a list or nparray

        self._patterns = np.empty((self._P, self._N))
        for i in range(len(patterns)):
            self._patterns[i] = patterns[i] if isinstance(patterns[i], list) else patterns[i].reshape(-1)

        if math.isinf(certainty):
            self._sigmoid = sgn
        else:
            def wrap(rand):
                def sig(x):
                    chance = (1 + math.tanh(rand * x)) / 2
                    return 1 if chance >= random() else -1
                return sig
            self._sigmoid = wrap(certainty)

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
                        weight_sum += self._patterns[k, i] * self._patterns[k, j]
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

    def compute(self, initial_state: np.ndarray, *, sync: bool = False) -> np.ndarray:
        '''
        Computes the stable configuration of the given state based on `weights`. Every pattern is a stable attractor, but they are not the only ones. 
        Negative initial patterns are also stable attractors, along with states that are equidistant to other attractors (with respect to Hamming Distance).

        `initial_state` can be of any shape; this method will automatically reshape everything.

        ### Sync Parameter

        By default, the pattern is computed asynchronously. More specifically, at the `i`th step, the `i mod N`
        node is updated, and this new pattern is used for the `(i + 1)` step. 
        (E.g. On Step 0, the 0th node is updated; this new pattern is used for Step 1 to update Node 1).

        If synchronous behavior is desired (i.e. all nodes are updated on the same step, using the same pattern),
        then pass `sync = True`. If `P` is close enough to `N` (see attributes), 
        this may result in an infinite updating loop.

        ### Certainty Hyperparameter

        By default, the state of a neuron is determined like so:
        ```
        State of neuron_{i} = sgn(sum of weight_{ij} * neuron_{j})
        ```
        If the certainty parameter was given in `__init__`, then the following formula will be used instead:
        ```
        Probability(neuron_{i} = 1) = (1 + math.tanh(certainty * sum)) / 2
        ```
        And of course, `P(neuron_{i} = -1) = 1 - P(neuron_{i} = 1)`. 

        ### Raises 
        `ValueError`: The given state is not the same size as the patterns.
        '''
        original_shape = initial_state.shape
        initial_state = initial_state.reshape(-1)

        if len(initial_state) != self.N: 
            raise ValueError(f"Invalid input size given. Expected {self.N}, got {len(initial_state)}")
        previous = np.zeros(1)
        current = initial_state

        while not np.array_equal(previous, current):
            previous = current
            this_step = current.copy() if sync else current # if sync is true, update a copy instead
            for i in range(len(current)):
                this_step[i] = self._sigmoid(np.dot(self.weights[i, :], current))
            current = this_step

        return current.reshape(original_shape)
    
    def is_learned_pattern(self, pattern: list[int] | np.ndarray[np.int8]) -> int:
        '''
        Checks if the given pattern is one of the initial trained patterns.

        Returns 1 if the given pattern is an initial pattern, -1 if it's the opposite (negative)
        of an initial pattern, and 0 otherwise. Note that this will not check for other spurious states.
        '''
        for i in range(self.P):
            check = self.patterns[i, :] # grabs the ith row
            if np.array_equal(pattern, check):
                return 1
            elif np.array_equal(pattern, -1 * check):
                return -1
            
        return 0

def create_space(N: int) -> list[np.ndarray[np.int8]]:
    '''
    Creates `{-1, 1}^N`

    Intended to run tests with `Network`.
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
                space.append(np.array(convert(comb + [x]), dtype=np.int8))
            else:
                looper(comb + [x])
    
    looper([])
    
    return space

