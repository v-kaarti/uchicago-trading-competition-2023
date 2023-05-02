"""
This file contains code for weighting a portfolio of stocks based on historical price data 
"""
import pandas as pd
import numpy as np
import scipy

history = pd.DataFrame(columns=[i for i in range(10)])

risk_factor = 6
return_factor = 1
lookback = 2600


def calculate_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    p_var, p_ret = calculate_performance(weights, mean_returns, cov_matrix)
    if p_ret > 0:
        return -np.power(p_ret, return_factor) / np.power(p_var, risk_factor)
    else:
        return -p_ret / np.power(p_var, risk_factor)


def allocate_portfolio(asset_prices):
    history.loc[history.shape[0]] = asset_prices
    curr_df = history.iloc[-min(lookback, history.shape[0]) :]

    num_assets = len(curr_df.columns)
    returns = curr_df.pct_change()
    mean_returns = curr_df.pct_change().mean()
    cov_matrix = returns.cov()

    args = (mean_returns, cov_matrix)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = scipy.optimize.minimize(
        neg_sharpe_ratio,
        num_assets
        * [
            1.0 / num_assets,
        ],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    max_sharpe = result.x
    return max_sharpe


def grading(
    testing,
):
    weights = np.full(shape=(len(testing.index), 10), fill_value=0.0)
    for i in range(0, len(testing)):
        unnormed = np.array(allocate_portfolio(list(testing.iloc[i, :])))
        positive = np.absolute(unnormed)
        normed = positive / np.sum(positive)
        weights[i] = list(normed)
    capital = [1]
    for i in range(len(testing) - 1):
        shares = capital[-1] * np.array(weights[i]) / np.array(testing.iloc[i, :])
        capital.append(
            float(
                np.matmul(np.reshape(shares, (1, 10)), np.array(testing.iloc[i + 1, :]))
            )
        )
    returns = (np.array(capital[1:]) - np.array(capital[:-1])) / np.array(capital[:-1])
    return np.mean(returns) / np.std(returns) * (252**0.5), capital, weights
