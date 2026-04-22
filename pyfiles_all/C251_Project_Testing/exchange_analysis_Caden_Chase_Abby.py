"""
exchange_analysis.py
Analyze a policy for buying and selling stocks.

Translated from exchange_analysis.m (MATLAB) by Henri P. Gavin.

INPUT (design_var):
    design_var[0] = q1      .... quality velocity coeff
    design_var[1] = q2      .... quality acceleration coeff
    design_var[2] = q3      .... quality volatility coeff
    design_var[3] = fc      .... fraction of cash to invest (0 < fc < 1)
    design_var[4] = phi     .... forgetting factor (0 < phi < 1)
    design_var[5] = B       .... buy  threshold on day 1
    design_var[6] = S       .... sell threshold on day 1
    design_var[7] = B_sens  .... buy sensitivity
    design_var[8] = S_sens  .... sell sensitivity

OUTPUT:
    cost        .... negative of total portfolio value (cash + investments) after 200 days
    constraint  .... always -1 (feasibility placeholder)
"""

import numpy as np
import matplotlib.pyplot as plt


def exchange_analysis(design_var, constants):
    """
    Simulate a stock trading strategy and return the negative total portfolio value.

    Parameters
    ----------
    design_var : array-like, length >= 11
        [q1, q2, q3, q4, q5, fc, phi, B, S, B_sens, S_sens]
    constants : list
        [FigNo, stock_prices]
        FigNo        : int  — if > 0, generate plots; if 0, skip plots
        stock_prices : 2D numpy array, shape (days, stocks)

    Returns
    -------
    cost       : float  — negative of (final investment value + cash)
    constraint : float  — always -1
    """

    # ------------------------------------------------------------------ design_vars
    q   = np.array(design_var[0:3], dtype=float)
    fc  = float(np.clip(design_var[3], 0.0, 1.0))   # fraction of cash to invest
    phi = float(np.clip(design_var[4], 0.0, 1.0))   # forgetting factor
    B   = float(design_var[5])                        # buy  threshold on day 1
    S   = float(design_var[6])                        # sell threshold on day 1
    
    B_sens = float(design_var[7])
    S_sens = float(design_var[8])

    FigNo        = constants[0]
    stock_prices = np.array(constants[1], dtype=float)  # (days, stocks)

    days, stocks = stock_prices.shape

    # --------------------------------------------------------- initialise arrays
    smooth_price = stock_prices.copy()           # running-average price
    smooth_veloc = np.zeros((days, stocks))      # fractional change of smooth price
    smooth_accel = np.zeros((days, stocks))      # rate of fractional change
    variance     = np.zeros((days, stocks))      # running variance of smooth_veloc

    quality      = np.zeros((days, stocks))      # quality metric Q_{d,s}
    B_record     = np.zeros(days)                # history of buy  threshold
    S_record     = np.zeros(days)                # history of sell threshold

    shares_owned = np.zeros((days, stocks))      # shares held each day
    value        = np.zeros(days)                # total investment value
    cash         = np.zeros(days)                # cash in hand

    cash[0:3]     = 1000.00                      # starting cash ($1000)
    B_record[0:3] = B
    S_record[0:3] = S

    buy_threshold    = B
    sell_threshold   = S
    transaction_cost = 2.00                      # $2 per transaction

    # ----------------------------------------------------------------- main loop
    for day in range(1, days - 1):

        d    = day      # today     (0-indexed)
        d_p1 = day + 1  # tomorrow
        d_m1 = day - 1  # yesterday

        # ----------------------------------------------------------
        # Running averages — DO NOT EDIT (equations 3-6 in spec)
        # ----------------------------------------------------------
        smooth_price[d_p1, :] = (
            (1 - phi) * smooth_price[d, :]
            + phi * stock_prices[d_p1, :]
        )

        denom = smooth_price[d_p1, :] + smooth_price[d_m1, :]
        denom = np.where(denom == 0, 1e-10, denom)   # guard against /0
        smooth_veloc[d_p1, :] = (
            (smooth_price[d_p1, :] - smooth_price[d_m1, :]) / denom
        )

        smooth_accel[d_p1, :] = (
            (smooth_veloc[d_p1, :] - smooth_veloc[d_m1, :]) / 2.0
        )

        variance[d_p1, :] = (
            (1 - phi) * variance[d, :]
            + phi * smooth_veloc[d_p1, :] ** 2
        )

        # Portfolio value today — DO NOT EDIT
        owned_idx = np.where(shares_owned[d, :] > 0)[0]
        value[d]  = np.sum(
            stock_prices[d, owned_idx] * shares_owned[d, owned_idx]
        )

        # --- EDIT #1: quality metric (standard equation 1) ---
        
        quality[d, :] = (
            q[0] * smooth_veloc[d, :]           # q1 * p_dot  (velocity)
            + q[1] * smooth_accel[d, :]         # q2 * p_ddot (acceleration)
            + q[2] * np.sqrt(variance[d, :])    # q3 * v      (volatility)
            #+ q[3] * np.sqrt(variance[d, :]) * smooth_veloc[d, :] # q4 * v * p_dot
            #+ q[4] * np.sqrt(variance[d, :]) * smooth_accel[d, :] # q5 * v * p_ddot
        )

        # --- EDIT #2: trading decisions (MODIFIED - bottom 10 ranking) ---
        # Identify the "Bottom 10" ranking threshold
        sorted_quality = np.sort(quality[d, :])
        bottom_10_val = sorted_quality[9]

        # Sell if quality is below threshold OR if it's in the bottom 5
        potentially_sellable = np.where((quality[d, :] < sell_threshold) | (quality[d, :] <= bottom_10_val))[0]
        potentially_buyable  = np.where((quality[d, :] > buy_threshold) & (quality[d, :] > bottom_10_val))[0]

        # No shorting — can only sell stocks you actually own
        owned_set = set(owned_idx.tolist())
        sell_set  = set(potentially_sellable.tolist())
        stocks_to_sell = np.array(sorted(owned_set & sell_set), dtype=int)

        # Don't sell if there is nothing to buy
        if len(potentially_buyable) == 0:
            stocks_to_sell = np.array([], dtype=int)

        # Execute sells — DO NOT EDIT
        if len(stocks_to_sell) > 0:
            cash[d] += np.sum(
                stock_prices[d, stocks_to_sell]
                * shares_owned[d, stocks_to_sell]
            )
            cash[d] -= transaction_cost * len(stocks_to_sell)
            shares_owned[d, stocks_to_sell] = 0.0

        # --- EDIT #3: execute buys (MODIFIED - transaction cost handling) ---
        shares_to_buy = np.zeros(stocks)
        stocks_to_buy = potentially_buyable

        if len(stocks_to_buy) > 0 and cash[d] > 0:
            total_buy_fees = transaction_cost * len(stocks_to_buy)
            if cash[d] > total_buy_fees:
                cash[d] -= total_buy_fees
                cash_to_invest = max(0.0, fc * cash[d])
                
                # Equal allocation across all stocks meeting the rank/threshold criteria
                individual_allocation = cash_to_invest / len(stocks_to_buy)
                shares_to_buy[stocks_to_buy] = individual_allocation / stock_prices[d, stocks_to_buy]
                cash[d] -= cash_to_invest

        # Carry shares and cash forward — DO NOT EDIT
        shares_owned[d_p1, :] = shares_owned[d, :] + shares_to_buy
        cash[d_p1] = cash[d] * (1.0 + 0.04 / 365.0)   # 4% annual safe return
        
        # --- EDIT #4: threshold update (MODIFIED - sensitivity parameters) ---
        owned_tomorrow = np.where(shares_owned[d_p1, :] > 0)[0]

        if len(owned_tomorrow) == 0:
            # No stocks held tomorrow — keep thresholds unchanged
            bb = buy_threshold
            ss = sell_threshold
        else:
            # Shift thresholds slightly based on sensitivity variables
            bb = np.max(quality[d, owned_tomorrow]) + B_sens
            ss = np.min(quality[d, owned_tomorrow]) - S_sens

        buy_threshold  = bb
        sell_threshold = ss

        # --- EDIT #5: minimum threshold gap (unchanged) ---
        sell_threshold = min(
            buy_threshold - 0.10 * abs(buy_threshold),
            sell_threshold
        )

        B_record[d] = buy_threshold
        S_record[d] = sell_threshold

    # --- final day value ---
    owned_idx       = np.where(shares_owned[days - 1, :] > 0)[0]
    value[days - 1] = np.sum(
        stock_prices[days - 1, owned_idx]
        * shares_owned[days - 1, owned_idx]
    )

    cost       = -(value[days - 1] + cash[days - 1])
    constraint = -1

    # ------------------------------------------------------------------ plotting
    if FigNo > 0:
        plt.ion()
        stp  = [0, 11, 18]              # stocks to plot (0-indexed = stocks 1, 12, 19)
        time = np.arange(1, days + 1)

        fig1, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        fig1.suptitle(f"design_vars = {np.round(design_var, 4)}", fontsize=9)

        axes[0].plot(time, stock_prices[:, stp], '-b', linewidth=0.8)
        axes[0].plot(time, smooth_price[:, stp], linewidth=0.8)
        axes[0].set_ylabel('price')

        axes[1].plot(time, smooth_veloc[:, stp], linewidth=0.8)
        axes[1].set_ylabel('% change')

        axes[2].plot(time, smooth_accel[:, stp], linewidth=0.8)
        axes[2].set_ylabel('change of % change')

        axes[3].plot(time, variance[:, stp], linewidth=0.8)
        axes[3].set_ylabel('variance')

        axes[4].plot(time, B_record, '--k', linewidth=1)
        axes[4].plot(time, S_record, '--k', linewidth=1)
        axes[4].plot(time, quality[:, stp], linewidth=0.8)
        axes[4].set_ylabel('quality')
        axes[4].set_xlabel('trading day')

        plt.tight_layout()
        plt.savefig(f'exchange_analysis_fig{FigNo+1}.png', dpi=150)

        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig2.suptitle(f"design_vars = {np.round(design_var, 4)}", fontsize=9)

        investment_values = shares_owned * stock_prices
        axes2[0].plot(time, investment_values, '-', linewidth=0.6)
        axes2[0].plot(time, value, '-b', linewidth=3, label='investments')
        axes2[0].plot(time, cash,  '-g', linewidth=3, label='cash')
        axes2[0].legend(loc='upper left')
        axes2[0].set_ylabel('value ($)')

        axes2[1].plot(time, shares_owned, '-', linewidth=0.8)
        axes2[1].set_ylabel('shares owned')
        axes2[1].set_xlabel('trading day')

        plt.tight_layout()
        plt.savefig(f'exchange_analysis_fig{FigNo+2}.png', dpi=150)

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(time, B_record, label='buy threshold')
        ax3.plot(time, S_record, label='sell threshold')
        ax3.set_ylabel('threshold values')
        ax3.set_xlabel('trading day')
        ax3.legend()
        plt.tight_layout()
        plt.savefig(f'exchange_analysis_fig{FigNo+3}.png', dpi=150)

        print(f"  Final value: ${-cost:.2f}  (investments: ${value[days-1]:.2f}, cash: ${cash[days-1]:.2f})")

    return cost, constraint