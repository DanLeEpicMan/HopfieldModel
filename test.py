import numpy as np
import altair as alt, pandas as pd
from network import Network


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
                space.append(np.array(convert(comb + [x]), dtype=np.int8))
            else:
                looper(comb + [x])
    
    looper([])
    
    return space

space = create_space(6)
data = {}

net = Network(patterns=np.array([
    [1, 1, 1, -1, 1, 1],
    [-1, 1, -1, -1, 1, -1]
], dtype=np.int8))

for config in space:
    stable = net.compute(config)
    key=str(stable)
    data[key] = data.get(key, 0) + 1

new_data = pd.DataFrame.from_dict({
    "Attractors": data.keys(),
    "#": data.values()
})

hist = alt.Chart(new_data).mark_bar(size=30).encode(
    x="Attractors",
    y="#"
).properties(
    width=250
).configure_axisX(labelAngle=45, labelColor='gray')

hist.show()
