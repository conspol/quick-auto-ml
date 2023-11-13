import numpy as np
from sklearn.covariance import (
    EmpiricalCovariance,
    LedoitWolf,
    MinCovDet,
    ShrunkCovariance,
)


class CovarianceMethod:
    def __init__(self, data):
        pass

    def fit(self, data):
        pass

    def get_covariance(self):
        pass


class ShrunkCov(CovarianceMethod):
    def __init__(self, data):
        self.lw_shrinkage = LedoitWolf().fit(data)
        self.model = ShrunkCovariance(shrinkage=self.lw_shrinkage.shrinkage_
                                      ).fit(data)
    
    def get_covariance(self):
        return self.model.covariance_


class MinCov(CovarianceMethod):
    def __init__(self, data):
        self.model = MinCovDet().fit(data)
    
    def get_covariance(self):
        return self.model.covariance_


class EmpCov(CovarianceMethod):
    def __init__(self, data):
        self.model = EmpiricalCovariance().fit(data)
    
    def get_covariance(self):
        return self.model.covariance_


class NumpyCov(CovarianceMethod):
    def __init__(self, data):
        self.data = data
        self.covariance_matrix = np.cov(data, rowvar=False)  # Assuming data has features as columns and samples as rows

    def get_covariance(self):
        return self.covariance_matrix
