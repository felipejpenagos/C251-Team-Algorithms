import numpy as np
from exchange_analysis_David_Erin_Sophia import exchange_analysis  # Using original, unmodified version

# Their reported optimal params
design_var = [-8.454427, 4.7047, 22.6595, 0.9664, 0.0776, 0.5913, 0.1794]

print("="*70)
print("STOCK_PRICES1 - FULL 200 DAYS (what they reported)")
print("="*70)
stock_prices1 = np.genfromtxt("stock_prices1.csv", delimiter=",")
cost1_full, _ = exchange_analysis(design_var, [1, stock_prices1])
print(f"Full 200 days: ${-cost1_full:.2f}\n")

print("="*70)
print("STOCK_PRICES2 - DAYS 201-400 (unseen grading data)")
print("="*70)
stock_prices2 = np.genfromtxt("stock_prices2.csv", delimiter=",")
cost2, _ = exchange_analysis(design_var, [20, stock_prices2])
print(f"Days 201-400: ${-cost2:.2f}\n")

print("="*70)
print("GRADING CALCULATION")
print("="*70)
print(f"Days 1-200 score:   {-cost1_full/200:.2f} / 10  (total return: {((-cost1_full/1000 - 1)*100):+.1f}%)")
print(f"Days 201-400 score: {-cost2/100:.2f} / 10  (total return: {((-cost2/1000 - 1)*100):+.1f}%)")