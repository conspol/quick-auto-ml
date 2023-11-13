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
from quick_auto_ml.data_proc import (
    get_num_features_data,
    load_dataframe,
    prepare_data,
    prepare_test_train,
)
from quick_auto_ml.defines import CLASS_LABEL
from quick_auto_ml.plots import show_feature_matrix
from quick_auto_ml.utils import (
    process_random_seeds,
    process_same_file_cfgs,
    register_base_config
)

config_store = ConfigStore.instance()
config_store.store(name='base_config', node=AppConfig)


def setup_initial(
    cfg: AppConfig,
) -> None:
    lg.debug(f"\n{OmegaConf.to_yaml(cfg)}")

    lg.configure(handlers=[{
        'sink': sys.stdout,
        'level': cfg.log_level.name,
    }])

    # ============= Prepare configs =============

    if cfg.change_nested_random_seeds:
        process_random_seeds(cfg)

    cfg_ds = cfg.data
    process_same_file_cfgs(cfg_ds)

    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")
    
    input_file_path = Path(cfg_ds.input_file)

    if (not cfg_ds.input_file_type.lower() == 'excel' 
        or not input_file_path.suffix == '.xlsx'):
        raise NotImplementedError(
            "Only xlsx input files are supported at this time. "
        )


def setup_data(cfg):
    cfg_ds = cfg.data

    data = load_dataframe(cfg_file=cfg_ds)

    lg.debug("Loaded dataframe:")
    lg.debug(f"\n{data}")

    data = prepare_data(data=data, cfg_ds=cfg_ds)

    lg.debug("Processed dataframe:")
    lg.debug(f"\n{data}")

    train_df, test_df = prepare_test_train(
        data=data,
        cfg_ds=cfg_ds,
    )

    return data, (train_df, test_df)


@hydra.main(
    config_path='quick_auto_ml/conf',
    config_name="_private_config",
    version_base=None,
)
def main(
    cfg: AppConfig,
) -> None:
    setup_initial(cfg)
    data, (train_df, test_df) = setup_data(cfg)

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