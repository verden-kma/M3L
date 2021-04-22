import numpy as np


def k_means(observations: np.ndarray, k: int):
    obs_num, obs_len = observations.shape
    centroids = np.empty((k, obs_len))
    obs_step = int(obs_num / k)

    prev_clusters = None
    while True:
        clusters = dict()
        for i in range(k):
            centroids[i] = observations[i * obs_step]
        for obsInd in range(obs_num):
            for centInd in range(k):
                dist = np.linalg.norm(centroids[centInd]-observations[obsInd])
                if (obsInd not in clusters):
                    clusters[obsInd] = (dist, centInd)
                else:
                    if (clusters[obsInd][0] > dist):
                        clusters[obsInd] = (dist, centInd)
        if (prev_clusters is not None) and (clusters == prev_clusters):
            break
        centroids = np.empty((k, obs_len))

        grouped_by_centroid = dict()
        for obsInd in range(obs_num):
            if (clusters[obsInd][1] not in grouped_by_centroid):
                grouped_by_centroid[clusters[obsInd][1]] = np.empty((1, 0))
            grouped_by_centroid[obsInd] = np.append(grouped_by_centroid[clusters[obsInd][1]], observations[obsInd])
        for centInd in range(k):
            centroids[centInd] = np.mean(grouped_by_centroid[0])
        prev_clusters = clusters
    return list(map(lambda x: x[1], list(prev_clusters.values())))
