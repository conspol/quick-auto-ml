import math
import sys
from pathlib import Path
from typing import Dict, List, Union

import h2o
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
from hydra.core.config_store import ConfigStore
from loguru import logger as lg
from omegaconf import OmegaConf

from quick_auto_ml.conf_schema.structured_configs import (
    AppConfig,
)
from quick_auto_ml.covariances import (
    CovarianceMethod,
    EmpCov,
    MinCov,
    NumpyCov,
    ShrunkCov,
)
from quick_auto_ml.defines import CLASS_LABEL
from quick_auto_ml.plots import show_feature_matrix
from quick_auto_ml.utils import (
    setup_initial,
    setup_data,
    register_base_config,
)

config_store = ConfigStore.instance()
config_store.store(name='base_config', node=AppConfig)




@hydra.main(
    config_path='quick_auto_ml/conf',
    config_name="_private_config",
    version_base=None,
)
def main(
    cfg: AppConfig,
) -> None:
    setup_initial(cfg)
    data, (train_df, test_df) = setup_data(cfg.data)

    h2o.init(
        nthreads=10,
        max_mem_size='12G',
    )

    hf = h2o.H2OFrame(train_df)
    test_hf = h2o.H2OFrame(test_df)
    exclude_cols = {cfg.data.label_column, CLASS_LABEL, *cfg.train.exclude_cols}
    feature_cols = [col_ for col_ in train_df.columns if col_ not in exclude_cols]

    nfolds = hf.nrows
    aml = H2OAutoML(
        max_models=50,
        max_runtime_secs=60,
        seed=cfg.random_seed,
        nfolds=nfolds,
    )

    lg.info("fin.")


if __name__ == '__main__':
    register_base_config()
    main()