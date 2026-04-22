import numpy as np
from exchange_analysis_Cory_Luke_Oliver import exchange_analysis

design_var = [-21.5105770, 28.9524825, 29.4495586, 0.678916666, 0.392263444, 0.0593436726, -0.00479652769]

print("="*70)
print("STOCK_PRICES1 - First 200 DAYS (Results match what they reported)")
print("="*70)
stock_prices1 = np.genfromtxt("stock_prices1.csv", delimiter=",")
cost1_full, _ = exchange_analysis(design_var, [1, stock_prices1])
print(f"Full 200 days: ${-cost1_full:.2f}\n")

# print("="*70)
# print("STOCK_PRICES2 - DAYS 201-400 (unseen grading data)")
# print("="*70)
# stock_prices2 = np.genfromtxt("stock_prices2.csv", delimiter=",")
# cost2, _ = exchange_analysis(design_var, [20, stock_prices2])
# print(f"Days 201-400: ${-cost2:.2f}\n")

