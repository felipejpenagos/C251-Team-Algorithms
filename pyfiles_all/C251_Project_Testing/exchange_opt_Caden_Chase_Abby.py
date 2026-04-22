import numpy as np
import matplotlib.pyplot as plt
from exchange_analysis_Caden_Chase_Abby import exchange_analysis
from multivarious.opt import nms

def objective_function(x, windows):
    # Physical Constraint (buy > sell)
    if x[6] >= (x[5] - 0.10 * abs(x[5])):
        return 2e6, 0.0
    
    # Run 50 day windows
    costs = []
    for w_data in windows:
        cost, _ = exchange_analysis(x, [0, w_data])
        costs.append(cost)
    
    # Optimizer
    # If it didn't trade in any of the 4 windows, penalize heavily.
    # If the cash value is negative, also penalize heavily.
    for c in costs:
        if abs(c + 1000) < 5.0:
            return 1e6, 0.0
        if c > 0:
            return 1e7, 0.0
    
    # Avg windows
    return np.mean(costs), -1.0

def run_optimization(csv_file):
    stock_prices = np.genfromtxt(csv_file, delimiter=',')
    
    # Make 50 day chunks
    q1 = stock_prices[150:200, :]
    q2 = stock_prices[50:100, :]
    q3 = stock_prices[100:150, :]
    q4 = stock_prices[0:50, :]
    all_windows = [q1, q2, q3, q4]
    
    v_lb = np.array([-20.0, -80.0, -150.0, 0.70, 0.05, -5.0, -10.0, -3, -4.0])
    v_ub = np.array([ 20.0,  80.0,  150.0, 0.99, 0.50, 15.0,  10.0,  6,  3.0])
    
    v_base = np.array([-11.995107, 53.793901, 147.700367, 0.990000, 0.378398, 7.363392, 1.438171, 1.057898, -0.322635])
    
    best_v_overall = None
    best_f_overall = 2e6
    
    # optimization for loop
    num_starts = 5
    for i in range(num_starts):
        print(f"\n Start {i+1}/{num_starts}...")
        
        if i == 0:
            v_init = v_base
        else:
            # vary base guess by 10% of bound range
            noise = (v_ub - v_lb) * 0.10 * np.random.randn(9)
            v_init = np.clip(v_base + noise, v_lb, v_ub)
        
        results = nms(
            lambda x, _: objective_function(x, all_windows),
            v_init, v_lb, v_ub,
            [0, 1e-5, 1e-5, 1e-5, 3000, 1500]
        )
        
        v_opt, f_opt = results[0], results[1]
        print(f" Done. Avg Profit: ${(-f_opt - 1000):.2f}")
        
        if f_opt < best_f_overall:
            best_f_overall = f_opt
            best_v_overall = v_opt
            print("New Best")
    
    print("\n" + "="*45)
    print("Optimal V")
    formatted_v = ", ".join([f"{val:.6f}" for val in best_v_overall])
    print(f"v_opt = np.array([{formatted_v}])")
    print(f"Overall Average Net Profit: ${(-best_f_overall - 1000):.2f}")
    print("="*45)
    
    plt.ioff()
    for i, data in enumerate(all_windows):
        print(f"\nDisplaying Quarter {i+1}...")
        exchange_analysis(best_v_overall, [1, data])
    
    plt.show()

if __name__ == "__main__":
    run_optimization('stock_prices1.csv')