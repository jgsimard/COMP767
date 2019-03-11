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


class TileCoding1D(LinearApproximationFunction):
    def __init__(self, n_bins, n_tilings, bounds):
        self.n_bins = n_bins
        self.n_tilings = n_tilings
        self.tile_width = (bounds[1] - bounds[0]) / ((n_bins - 1) * n_tilings + 1) * n_tilings
        self.offset = self.tile_width / n_tilings

    def get_feature_vector(self, s):
        feature_vector = np.zeros(self.n_bins * self.n_tilings)
        for tiling in range(self.n_tilings):
            s_prime = s + tiling * self.offset
            bin_index = np.floor(s_prime / self.tile_width).astype(int)
            feature_vector[tiling * self.n_bins + bin_index] = 1
        return feature_vector

class TileCoding2D(LinearApproximationFunction):
    def __init__(self, n_bins, n_tilings, bounds_box):
        self.n_bins = n_bins
        self.n_tilings = n_tilings
        normalization_factor = n_tilings / ((n_bins - 1) * n_tilings + 1)
        self.tile_shape = np.array([bounds_box.high[0] - bounds_box.low[0],
                                    bounds_box.high[1] - bounds_box.low[1]]) * normalization_factor * 1.1 #if not problem in limit cases
        self.bounds = bounds_box
        self.offset = self.tile_shape / n_tilings

    def get_feature_vector(self, s):
        feature_vector = np.zeros(self.n_bins**2 * self.n_tilings)
        for tiling in range(self.n_tilings):
            s_prime = s + tiling * self.offset - self.bounds.low
            bin_index = np.floor(s_prime / self.tile_shape).astype(int)
            bin_index = np.ravel_multi_index(bin_index, (self.n_bins, self.n_bins))
            feature_vector[tiling * self.n_bins**2 + bin_index] = 1
        return feature_vector
