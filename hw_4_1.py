import numpy as np


def k_means(observations: np.ndarray, k: int):
    obs_num, obs_len = observations.shape
    centroids = np.empty((k, obs_len))
    obs_step = int(obs_num / k)

    prev_clusters = None
    for i in range(k):
        centroids[i] = observations[i * obs_step]
    while True:
        clusters = dict()
        for obsInd in range(obs_num):
            for centInd in range(k):
                dist = np.linalg.norm(centroids[centInd] - observations[obsInd])
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
                grouped_by_centroid[clusters[obsInd][1]] = np.array([])
            temp1 = grouped_by_centroid[clusters[obsInd][1]]
            temp2 = observations[obsInd]
            grouped_by_centroid[clusters[obsInd][1]] = np.concatenate((temp1, temp2), axis=0)
        for centInd in range(k):
            centroids[centInd] = np.mean(grouped_by_centroid[centInd])
        prev_clusters = clusters
    return list(map(lambda x: x[1], list(prev_clusters.values())))


test_obs = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3],
                     [101, 101, 101], [102, 102, 102], [103, 103, 103], [104, 104, 104],
                     [151, 151, 151], [152, 152, 152], [153, 153, 153], [154, 154, 154]])
k = 3
print(k_means(test_obs, k))