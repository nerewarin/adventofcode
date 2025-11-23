import matplotlib.pyplot as plt
import numpy as np

y = np.array(
    [
        8,
        11,
        13,
        19,
        29,
        45,
        72,
        105,
        152,
        227,
        353,
        538,
        818,
        1219,
        1791,
        2862,
        4337,
        6444,
        9820,
        14785,
        22897,
        34564,
        51981,
        79929,
        120193,
        183484,
        279703,
        421289,
        643983,
        974058,
        1481691,
        2259130,
        3408277,
        5201265,
        7892987,
        11965325,
        18239869,
        27606433,
        42011370,
        63828527,
        147381030,
        223396955,
        339412402,
        516161376,
        782634342,
        1190625569,
    ]
)

plt.plot(np.log(y))
plt.title("log(y)")
plt.xlabel("x")
plt.ylabel("log(y)")
plt.grid(True)
plt.show()

plt.plot(y)
plt.yscale("log")
plt.title("y with log-scale")
plt.xlabel("x")
plt.ylabel("y (log scale)")
plt.grid(True, which="both")
plt.show()

x = np.arange(len(y))
logy = np.log(y)

b, a = np.polyfit(x, logy, 1)

x_pred = 75
y_pred = np.exp(a + b * x_pred)
print(a, b, y_pred)
