import numpy as np
import altair as alt, pandas as pd
from network import Network, create_space


space = create_space(6)
data = {}

net = Network(patterns=np.array([
    [1, 1, 1, -1, 1, 1],
    [-1, 1, -1, -1, 1, -1],
    [-1, 1, 1, -1, 1, 1]
], dtype=np.int8))

def state_type_str(x: int) -> str:
    match x:
        case 1:
            return 'Initial'
        case -1:
            return 'Analog'
        case _:
            return 'Spurious'

for config in space:
    stable = net.compute(config)
    key=str(stable)
    data[key] = data.get(key, [stable, key, 0, state_type_str(net.is_learned_pattern(stable))])
    data[key][2] = data[key][2] + 1

data = pd.DataFrame(data).transpose()
data.columns = ['Pattern', 'Label', 'Value', 'Type']
data.index = np.arange(len(data.index))

hist = alt.Chart(data).mark_bar(size=30).encode(
    x=alt.X("Label", title='Pattern'),
    y=alt.Y("Value", title='# of Attracted States'),
    color=alt.Color("Type", scale=alt.Scale(scheme='set1'))
).properties(
    width=40*len(data.index)
).configure_axisX(labelAngle=-45, labelColor='gray')

hist.show()
