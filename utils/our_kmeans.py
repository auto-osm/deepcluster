from sklearn.cluster import KMeans

def run_our_kmeans(x_outs, k):
    n, _ = x_outs.shape
    kmeans = KMeans(n_clusters=k).fit(x_outs)

    flat_predictions = kmeans.labels_
    assert(flat_predictions.shape[0] == n)

    return flat_predictions, kmeans.inertia_, kmeans.cluster_centers_
