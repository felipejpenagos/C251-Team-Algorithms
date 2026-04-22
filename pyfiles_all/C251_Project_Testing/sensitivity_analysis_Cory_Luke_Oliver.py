import numpy as np
from exchange_analysis_Cory_Luke_Oliver import exchange_analysis

stock_prices = np.genfromtxt("stock_prices1.csv", delimiter=",")

def sensitivity_analysis(v_opt, stock_prices, names=None, pct=0.05):
    """
    Perform +5% sensitivity analysis on design variables.
    
    Parameters
    ----------
    v_opt : array
        Optimized design variables
    stock_prices : array
        Data to evaluate on (usually training data)
    names : list of str (optional)
        Variable names
    pct : float
        5% perturbation percentage
    
    Returns
    -------
    results : list of dict
        Sensitivity results for each variable
    """
    v_opt = np.array(v_opt)
    if names is None:
        names = ["q1", "q2", "q3", "fc", "phi", "B", "S"]
    
    # Baseline value
    cost_base, _ = exchange_analysis(v_opt, [0, stock_prices])
    base_value = -cost_base
    
    print("\n===== SENSITIVITY ANALYSIS =====")
    print(f"Baseline value: ${base_value:.2f}")
    
    results = []
    
    # Looping through values
    for i in range(len(v_opt)):
        v_test = v_opt.copy()
        # perturb by +5%
        v_test[i] *= (1 + pct)
        cost, _ = exchange_analysis(v_test, [0, stock_prices])
        value = -cost
        change = value - base_value
        percent_change = 100 * change / base_value if base_value != 0 else 0
        
        print(f"\nParameter: {names[i]}")
        print(f"  New value: {v_test[i]:.4f}")
        print(f"  New earnings: ${value:.2f}")
        print(f"  Change: ${change:.2f} ({percent_change:.2f}%)")
        
        results.append({
            "param": names[i],
            "value": value,
            "change": change,
            "percent_change": percent_change
        })
    
    # Identify the most sensitive
    most_sensitive = max(results, key=lambda x: abs(x["percent_change"]))
    print("\n===== MOST SENSITIVE PARAMETER =====")
    print(f"{most_sensitive['param']} ({most_sensitive['percent_change']:.2f}% change)")
    
    return results

v_best = [-21.5105770, 28.9524825, 29.4495586, 0.678916666, 0.392263444, 0.0593436726, -0.00479652769]

results = sensitivity_analysis(
    v_best,
    stock_prices
)