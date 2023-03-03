import numpy as np
import altair as alt, pandas as pd
from network import Network, create_space


space = create_space(6)
data = {}

net = Network(patterns=np.array([
    [1, 1, 1, -1, 1, 1],
    [-1, 1, -1, -1, 1, -1]
], dtype=np.int8))

def to_color(x: int) -> alt.Color:
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
    data[key] = data.get(key, [stable, 0, to_color(net.is_learned_pattern(stable))])
    data[key][1] = data[key][1] + 1

data = pd.DataFrame(data).transpose().rename(columns={0: 'Patterns', 1: 'Value', 2: 'Type'})

print(data)

hist = alt.Chart(data).mark_bar(size=30).encode(
    x="Patterns",
    y="Value",
    color="Type"
).properties(
    width=250
).configure_axisX(labelAngle=45, labelColor='gray')

hist.show()
