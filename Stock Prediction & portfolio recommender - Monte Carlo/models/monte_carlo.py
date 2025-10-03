import numpy as np
import pandas as pd
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Class for monte carlo simulation
    Using Geometric Brownian Motion(GBR):
    dS/S = μ dt + σ dW

    variables:
    - S = price of share
    - μ = drift (avg return)
    - σ = volatility (zmienność)
    - dW = Wiener process (Brownian motion)
    """

    def __init__(self,seed: int=42):
        """
        Args:
            prices: Series with historical prices
        Returns:
            Tuple (mu, sigma) - annual params
        """
        self.seed = seed
        self.mu = None
        self.sigma = None
        self.S0 = None

    def estimate_parameters(self,prices:pd.Series)-> Tuple[float,float]:
        """
        Estimating params μ (drift) and σ (volatility) from historic prices
        Args:
            prices: Series with historical prices(adj_close)
        Returns:
            Tuple (mu,sigma) - annual params
        """
        log_returns = np.log(prices/prices.shift(1)).dropna()

        daily_mu = log_returns.mean()
        daily_sigma = log_returns.std()

        #mu_annual = mu_daily * 252
        # sigma_annual = daily_sigma*252
        annual_mu = daily_mu * 252
        annual_sigma = daily_sigma * 252

        self.mu = annual_mu
        self.sigma = annual_sigma
        self.S0 = prices.iloc[-1]

        logger.info(f'Estimated params: mu = {self.mu}, sigma = {self.sigma}')
        logger.info(f'Starting price: {self.S0: .2f}')

        return annual_mu, annual_sigma

    def simulate(self,
                 S0: float,
                 mu: float,
                 sigma: float,
                 T: int,
                 n_simulations: int = 10000,
                 dt: float = 1/252)-> np.ndarray:
        """
        Simulation using Monte Carlo simulation
        Args:
            S0: starting price
            mu: Drift(annual)
            sigma: volatility(annual)
            T: time
            n_simulations: number of simulations
            dt: time step

        """
        np.random.seed(self.seed)

        logger.info(f'Starting {n_simulations} simulations on {T} days')
        n_steps = T
        Z = np.random.standard_normal((n_simulations,n_steps))
        drift = (mu-0.5*sigma**2)*dt
        diffusion = sigma*np.sqrt(dt)*Z
        log_returns = drift + diffusion
        cumulative_log_returns = np.cumsum(log_returns,axis=1)

        price_paths = S0* np.exp(cumulative_log_returns)
        price_paths = np.column_stack([np.full(n_simulations,S0), price_paths])

        logger.info(f'Simulation finishe. Shape: {price_paths.shape}')

        return price_paths

    def get_statistics(self,price_paths: np.ndarray) -> Dict[str,np.ndarray]:
        """
        Calculating statistics
        Args:
            price_paths: Array

        Returns:
            Dict with statistics for each day
        """

        stats = {
            'mean': np.mean(price_paths, axis=0),
            'median': np.median(price_paths, axis=0),
            'stdev': np.std(price_paths, axis=0),
            'min': np.min(price_paths, axis=0),
            'max': np.max(price_paths, axis=0),
            'percentile_5': np.percentile(price_paths, 5, axis=0),
            'percentile_25': np.percentile(price_paths, 25, axis=0),
            'percentile_75': np.percentile(price_paths, 75, axis=0),
            'percentile_95': np.percentile(price_paths, 95, axis=0),
        }
        return stats

    def calculate_risk_metrics(self,
                               price_paths: np.ndarray,
                               confidence_level: float = 0.95)-> Dict[str,float]:

        """
        Calculate risk metrics
        Args:
            price_paths: Array with simulated paths
            confidence_level: confidence level

        Returns:
            Dict with risk metrics
        """

        S0 = price_paths[:,0][0]
        final_prices = price_paths[:,-1]

        returns = (final_prices - S0) / S0
        var=np.percentile(returns, (1-confidence_level)*100)
        es = returns[returns<=var].mean()
        prob_profit = (returns>0).mean()
        expected_return = returns.mean()
        metrics = {
            'expected_return':expected_return,
            'var_95':var,
            'expected_shortfall':es,
            'probability_profit':prob_profit,
            'std_return':returns.std(),
            'best_case':returns.max(),
            'worst_case': returns.min()
        }
        logger.info(f'Risk metrics:(Var 95%: {var:.2%})')
        return metrics

    def probability_above_target(self,
                                 price_paths: np.ndarray,
                                 target_prices:float,
                                 day: int=-1)-> float:
        """
        Calculate probability above target
        Args:
            price_paths: Array with simulated paths
            target_prices: Target prices
            day: Day number (-1 = last)

        Returns:
            probabilityu (0-1)
        """
        prices_at_day = price_paths[:,day]
        prob = (prices_at_day >= target_prices).mean()
        logger.info(f'P(price >= ${target_prices: .2f}) = {prob:.2f}')
        return prob



### Helping visualization
def plot_simulations(price_paths:np.ndarray,
                         stats: Dict[str,np.ndarray],
                         ticker: str,
                         n_paths_to_plot: int = 100):
        """

        Drawing simulations results Monte Carlo

        """
        import matplotlib.pyplot as plt
        n_days = price_paths.shape[1]
        days = np.arange(n_days)


        fig, ax = plt.subplots(figsize=(14,7))
        n_to_plot = min(n_paths_to_plot, price_paths.shape[0])
        for i in range(n_to_plot):
            ax.plot(days,price_paths[i],alpha=0.05,color='gray',linewidth=0.5)

        ax.plot(days,stats['median'],label='Median',color='blue',linewidth=2)
        ax.plot(days,stats['mean'],label='Mean',color='red',linewidth=2,linestyle='--')

        ax.fill_between(days,stats['percentile_5'],stats['percentile_95'],alpha=0.2,color='blue',label='90% CI (5%-95%)')
        ax.fill_between(days,stats['percentile_25'],stats['percentile_75'],alpha=0.3,color='blue',label='50% CI (25%-75%)')

        ax.set_title(f'{ticker} - Monte Carlo simulations ({price_paths.shape[0]} paths)', fontsize=16,fontweight='bold')

        ax.set_xlabel('Day')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_final_distribution(price_paths: np.ndarray, ticker: str, cut_outliers: bool = True):
    """
    Drawing final distribution Monte Carlo
    """
    import matplotlib.pyplot as plt
    import numpy as np

    final_prices = price_paths[:, -1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(final_prices, bins=100, alpha=0.7, edgecolor='black', density=True)
    ax.axvline(final_prices.mean(), color='green', linestyle='--',
               linewidth=2, label=f'Mean: ${final_prices.mean():.2f}')
    ax.axvline(np.median(final_prices), color='blue', linestyle='--',
               linewidth=2, label=f'Median: ${np.median(final_prices):.2f}')

    if cut_outliers:

        low, high = np.percentile(final_prices, [1, 99])
        ax.set_xlim(low, high)

    ax.set_title(f'{ticker} - Final price distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('End price ($)')
    ax.set_ylabel('Probability density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    print("🎲 Test Monte Carlo Simulator\n")

    simulator = MonteCarloSimulator(seed=42)

    S0 = 100  # Initial price
    mu = 0.10  # 10% annual return
    sigma = 0.20  # 20% annual volatility
    T = 252  # 1 year (252 trading days)
    n_sims = 10000

    paths = simulator.simulate(S0, mu, sigma, T, n_sims)

    stats = simulator.get_statistics(paths)
    print(f"Average final price: ${stats['mean'][-1]:.2f}")
    print(f"Median final price: ${stats['median'][-1]:.2f}")

    # Risk metrics
    risk = simulator.calculate_risk_metrics(paths)
    print(f"\nExpected return: {risk['expected_return']:.2%}")
    print(f"VaR (95%): {risk['var_95']:.2%}")
    print(f"Expected Shortfall: {risk['expected_shortfall']:.2%}")
    print(f"Probability of profit: {risk['probability_profit']:.2%}")

    # Probability of exceeding threshold
    target = 110
    prob = simulator.probability_above_target(paths, target)

    print(f"\n✅ Test completed!")

