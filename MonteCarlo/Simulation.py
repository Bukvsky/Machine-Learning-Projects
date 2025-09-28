import numpy as np
import matplotlib.pyplot as plt

So = 100 # INITIAL STOCK PRICE
mu = 0.07 # expected annual return
sigma = 0.2 # Volatility
T = 1 # time horizon
dt = 1/252 # Daily steps
N = int(T/dt) #num of steps
simulations = 5 # Number of simulated paths

for i in range(simulations):
    prices = [So]
    for _ in range(N):
        shock = np.random.normal(loc=(mu*dt), scale = (sigma*np.sqrt(dt)))
        prices.append(prices[-1]*np.exp(shock))

    plt.plot(prices)


plt.title("Monte Carlo Stock Price Simulation")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()