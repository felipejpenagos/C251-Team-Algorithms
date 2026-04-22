from exchange_analysis_David_Erin_Sophia import exchange_analysis
import numpy as np
from scipy.optimize import minimize, dual_annealing

stock_prices = np.genfromtxt("stock_prices1.csv", delimiter=",") # DO NOT CHANGE
if np.isnan(stock_prices[:, -1]).all():
    stock_prices = stock_prices[:, :-1]

constants = [0, stock_prices] # DO NOT CHANGE

PENALTY = 1e6

def objective(x):
    """added to ensure that Buy threshold is always greater than Sell threshold"""
    if x[5] <= x[6]:
        return PENALTY
    cost, _ = exchange_analysis(x, constants)
    return cost

# design_var: [q1, q2, q3, fc, phi, B, S]
bounds = [
    (-30, 30),   # q1
    (-30, 30),   # q2
    (-30, 30),   # q3
    (0.01, 1.0), # fc
    (0.01, 0.99),# phi
    (-1, 1),     # B
    (-1, 1),     # S
]

"""Used dual annealing to find a rough guess for the optimal parameters
"""
da_result = dual_annealing(
    objective, bounds, seed=42, maxiter=500, x0=[-10, 2, -2, 0.9, 0.5, 0.1, -0.1],
)

print(f" DA best: {np.round(da_result.x, 4)}")
print(f" Value: ${-da_result.fun:.2f}")
print(f" B={da_result.x[5]:.4f} > S={da_result.x[6]:.4f}: {da_result.x[5] > da_result.x[6]}")

"""
We use the best guess from dual annealing to seed the local search
"""
initial_guesses = [
    da_result.x
]

best_cost = np.inf
best_x = None

"""
We use both Powell's method and Nelder-Mead method to seach for the optimal parameters
with our rough guesses
"""
for i, x0 in enumerate(initial_guesses):
    tag = "DA result" if i == 0 else f"Guess {i}"
    for method in ["Powell", "Nelder-Mead"]:
        res = minimize(objective, x0, method=method, options={"maxiter": 10000})
        if res.fun < PENALTY and res.x[5] > res.x[6]:
            val = -res.fun
            if i == 0 or method == "Powell":
                print(f"\n {tag} ({method}): {np.round(np.asarray(x0), 4)}")
                print(f" Optimal: {np.round(res.x, 4)}")
                print(f" Value: ${val:.2f} B={res.x[5]:.4f} S={res.x[6]:.4f}")
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x.copy()
            x0 = res.x  # feed into next method

# Sensitivity analysis
labels = ["q1", "q2", "q3", "fc", "phi", "B", "S"]
base_val = -best_cost

# Write results to text file
with open("optimization_results.txt", "w") as f:
    f.write("BEST RESULT\n")
    f.write(f"Parameters: {np.round(best_x, 6)}\n")
    f.write(f" q1={best_x[0]:.4f} q2={best_x[1]:.4f} q3={best_x[2]:.4f}\n")
    f.write(f" fc={best_x[3]:.4f} phi={best_x[4]:.4f}\n")
    f.write(f" B={best_x[5]:.4f} S={best_x[6]:.4f} (B > S: {best_x[5] > best_x[6]})\n")
    f.write(f"Full value (200 days): ${-best_cost:.2f}\n")
    f.write("\n")
    f.write("SENSITIVITY ANALYSIS (+5% perturbation)\n")
    f.write(f"{'param':<6} {'optimal':>10} {'perturbed':>10} {'value($)':>10} {'change($)':>10}\n")
    for j in range(len(best_x)):
        x_pert = best_x.copy()
        x_pert[j] *= 1.05
        if x_pert[5] <= x_pert[6]:
            pert_val = 0.0
        else:
            pert_cost, _ = exchange_analysis(x_pert, constants)
            pert_val = -pert_cost
        f.write(f"{labels[j]:<6} {best_x[j]:>10.4f} {x_pert[j]:>10.4f} {pert_val:>10.2f} {pert_val - base_val:>+10.2f}\n")

# Generate figures
exchange_analysis(best_x, [20, stock_prices])