import numpy as np


class ApproximationFunction:
    def __call__(self, s, w):
        pass

    def grad(self, s, w):
        pass


class LinearApproximationFunction:
    def get_feature_vector(self, s):
        pass

    def  __call__(self, s, w):
        feature_vector = self.get_feature_vector(s)
        return  w.T @ feature_vector

    def grad(self, s, w):
        return self.get_feature_vector(s)


class TileCoding(LinearApproximationFunction):
    def __init__(self, n_bins, n_tilings, observation_space):
        self.dims = observation_space.shape[0]
        self.n_bins = n_bins
        self.n_tilings = n_tilings
        normalization = n_tilings / ((n_bins - 1) * n_tilings + 1)
        self.tile = np.array([high - low for high, low in zip(observation_space.high, observation_space.low)]) * normalization

        self.observation_space = observation_space
        self.offset = self.tile / n_tilings
        self.tiling_size = n_bins**self.dims
        self.size = self.tiling_size * n_tilings

    def get_feature_vector(self, s):
        feature_vector = np.zeros(self.size)
        for tiling in range(self.n_tilings):
            s_prime = s + tiling * self.offset - self.observation_space.low
            index_in_tiling = np.floor(s_prime / self.tile).astype(int)
            index_in_tiling[index_in_tiling == self.n_bins] = self.n_bins - 1 #for cases at the edge
            if len(index_in_tiling) > 1:
                index_in_tiling = np.ravel_multi_index(index_in_tiling, (self.n_bins,)*self.dims)
            feature_vector[tiling * self.tiling_size + index_in_tiling] = 1
        return feature_vector


if __name__ == "__main__":
    import gym
    low = np.array([0.0, 0.0])
    high = np.array([1.0, 1.0])
    observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    tile_coding = TileCoding(n_bins=2, n_tilings=2, observation_space=observation_space)
    ij = [0.1, 0.5, 0.9]
    for i in ij:
        for j in ij:
            print(i,j, tile_coding.get_feature_vector([i,j]))
    print(tile_coding.tile)