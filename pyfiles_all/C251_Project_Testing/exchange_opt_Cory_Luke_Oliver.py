import numpy as np
from multivarious.opt import ors, nms
from exchange_analysis_Cory_Luke_Oliver import exchange_analysis
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

stock_prices = np.genfromtxt("stock_prices1.csv", delimiter=",")

# training/test split
stock_prices_train = stock_prices[:100, :]
stock_prices_test = stock_prices[100:, :]

# Objective test functions
def trading_objective(v, C):
    q1, q2, q3, fc, phi, B, S = v
    cost, _ = exchange_analysis(v, [0, C])
    f = cost

    # Penalty for no trading
    initial_cash = 1000
    final_value = -cost
    penalty = 0
    if abs(final_value - initial_cash) < 50:
        penalty = 500
    f = f + penalty

    # Constraints
    g = np.array([
        -fc,
        fc - 1,
        -phi,
        phi - 1,
        (S - B) / 0.1
    ])
    return f, g

# Bounds
v_lb = np.array([-50, -50, -50, 0.01, 0.01, -0.1, -0.2])
v_ub = np.array([50, 50, 50, 0.99, 0.99, 0.2, 0.1])

# Options
n = len(v_lb)
opts = [
    0,
    1e-2,
    1e-2,
    1e-3,
    50 * n**3,
    0.7,
    0.5,
    1,
    0.05
]

def explore_and_filter(n_samples, threshold):
    good_results = []
    good_values = []

    print("\n===== EXPLORING + FILTERING =====")

    for i in range(n_samples):
        print(f"\n--- SAMPLE {i+1} ---")

        v_init = v_lb + np.random.rand(len(v_lb)) * (v_ub - v_lb)

        v_opt, f_opt, _, _, _, _ = nms(
            trading_objective,
            v_init,
            v_lb,
            v_ub,
            opts,
            stock_prices_train
        )

        value = -f_opt
        print(f"Training value: ${value:.2f}")

        if value > threshold:
            print(">>> ACCEPTED")
            good_results.append(v_opt)
            good_values.append(value)
        else:
            print("Rejected")

    return np.array(good_results), np.array(good_values)

def find_clusters(results, eps=1.0, min_samples=2):
    """
    results: array of shape (n_samples, n_variables)
    eps: clustering radius (after normalization)
    min_samples: minimum points per cluster
    """
    if len(results) == 0:
        print("No results to cluster.")
        return []

    # Normalizing data
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    # Avoiding dividing by zero
    std[std == 0] = 1
    results_norm = (results - mean) / std

    # Clustering using DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(results_norm)

    clusters = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue  # -1 = noise
        cluster_points = results[labels == label]
        clusters.append(cluster_points)

    return clusters

def analyze_clusters(clusters):
    print("\n===== IMPROVED CLUSTER ANALYSIS =====")

    if len(clusters) == 0:
        print("No clusters found.")
        return

    param_names = ["q1", "q2", "q3", "fc", "phi", "B", "S"]

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        print(f"\nCluster {i+1}")
        print(f"Size: {len(cluster)}")

        mean = np.mean(cluster, axis=0)
        std = np.std(cluster, axis=0)

        for j in range(len(param_names)):
            print(f"{param_names[j]}: mean={mean[j]:.3f}, std={std[j]:.3f}")

def refine_clusters(clusters):
    refined_results = []

    print("\n===== REFINING CLUSTERS (USING MEAN) =====")

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        v_init = np.mean(cluster, axis=0)

        print(f"\n--- Cluster {i+1} ---")
        print("Cluster size:", len(cluster))
        print("Mean initial guess:", v_init)

        # Run ORS from cluster mean
        v_opt, f_opt, _, _, _, _ = ors(
            trading_objective,
            v_init,
            v_lb,
            v_ub,
            opts,
            stock_prices_train
        )

        train_value = -f_opt

        # Test performance
        cost_test, _ = exchange_analysis(v_opt, [0, stock_prices_test])
        test_value = -cost_test

        print(f"Train: ${train_value:.2f}, Test: ${test_value:.2f}")

        refined_results.append({
            "cluster": i + 1,
            "v_init": v_init,
            "v_opt": v_opt,
            "train": train_value,
            "test": test_value,
            "size": len(cluster)
        })

    return refined_results

def best_solution(refined_results):
    best = max(refined_results, key=lambda x: x["test"])

    print("\n===== BEST TEST SOLUTION =====")
    print("Cluster:", best["cluster"])
    print("Design vars:", best["v_opt"])
    print(f"Train value: ${best['train']:.2f}")
    print(f"Test value: ${best['test']:.2f}")

    return best

def plot_clusters(results, clusters):
    if len(results) == 0:
        print("No data to plot.")
        return

    # Normalize
    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    std[std == 0] = 1
    results_norm = (results - mean) / std

    # Creating PCA graphs
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(results_norm)

    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)

        # Find indices of these points
        idx = []
        for v in cluster:
            for j, r in enumerate(results):
                if np.allclose(v, r):
                    idx.append(j)

        cluster_points = points_2d[idx]

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=colors[i % len(colors)],
            label=f"Cluster {i+1}"
        )

    plt.title("Cluster Visualization (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(f'ClusterPlot.png', dpi=150)
    plt.show()

good_results, good_values = explore_and_filter(
    n_samples=2000,
    threshold=1000
)

clusters = find_clusters(good_results, eps=0.75, min_samples=4)
analyze_clusters(clusters)
plot_clusters(good_results, clusters)

refined_results = refine_clusters(clusters)
best = best_solution(refined_results)

print("\nRunning best solution on test data...\n")
exchange_analysis(best["v_opt"], [1, stock_prices_test])