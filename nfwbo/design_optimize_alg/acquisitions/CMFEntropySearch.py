import numpy as np

from design_optimize_alg.acquisitions.EntropySearch import EntropySearch
from design_optimize_alg.acquisitions.ExpectedImprovement import ExpectedImprovement


class CMFEntropySearch(EntropySearch):
    '''
    https://arxiv.org/abs/1605.07079
    Eq.7
    '''
    def __init__(self, gps, config, logger=None):
        super().__init__(gps, config, logger=logger)

    def configure(self):
        config = self.config
        self.num_representer_points = config["num_representer_points"]
        self.num_samples = config["num_samples"]
        self.target_fidelity_index = config["target_fidelity_index"]
        self.highest_fidelity = config["highest_fidelity"]
        self._config_sampler()
        self._config_proposal_function()

    def _config_proposal_function(self):
        bounds = self.config['sampler']['bounds']
        ei = ExpectedImprovement(self.gps, self.logger)

        def proposal_func(x):
            x_ = x[None, :]
            # Map to highest fidelity since only the information gain about the highest_fidelity is considered
            idx = np.ones((x_.shape[0], len(self.target_fidelity_index))) * self.highest_fidelity
            x_ = np.insert(x_, [self.target_fidelity_index[0]], idx, axis=1)
            
            val = np.log(np.clip(ei.evaluate(x_)[0], 0., np.PINF))
            if np.any(np.isnan(val)):
                return np.array([np.NINF])
            elif np.all([np.greater_equal(x_[:, :x.shape[0]], bounds[0, :]), np.greater_equal(bounds[1, :], x_[:, :x.shape[0]])]):
                return val
            else:
                return np.array([np.NINF])

        self.proposal_func = proposal_func

    def _sample_representer_points(self):
        repr_points, repr_points_log = super()._sample_representer_points()
        # Add fidelity index to representer points
        #idx = np.ones((repr_points.shape[0])) * self.highest_fidelity
        #repr_points = np.insert(repr_points, self.target_fidelity_index, idx, axis=1)
        idx = np.ones((repr_points.shape[0], len(self.target_fidelity_index))) * self.highest_fidelity
        repr_points = np.insert(repr_points, [self.target_fidelity_index[0]], idx, axis=1)

        return repr_points, repr_points_log


if __name__ == "__main__":
    params_acq = {}
    params_acq["name"] = "CMF-ES"
    params_acq["num_representer_points"] = 50
    params_acq["num_samples"] = 100
    params_acq["target_fidelity_index"] = 2
    params_acq["highest_fidelity"] = 50000
    config = params_acq
    es = CMFEntropySearch(None, config)

