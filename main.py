import sys
from pathlib import Path

import h2o
import hydra
import pandas as pd
from h2o.automl import H2OAutoML
from hydra.core.config_store import ConfigStore
from loguru import logger as lg
from omegaconf import OmegaConf

from quick_auto_ml.conf_schema.structured_configs import AppConfig
from quick_auto_ml.data_proc import process_num_df_to_binaryclass


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
    lg.info(OmegaConf.to_yaml(cfg))

    lg.configure(handlers=[{
        'sink': sys.stdout,
        'level': cfg.log_level.name,
    }])

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

    cfg_ds = cfg.data

    input_file_path = Path(cfg_ds.input_file)

    if not cfg_ds.input_file_type == 'xlsx' or not input_file_path.suffix == '.xlsx':
        raise NotImplementedError(
            "Only xlsx input files are supported at this time. "
        )

    data=pd.read_excel(
        input_file_path,
        index_col=0,
        sheet_name=cfg_ds.input_file_sheet_name,
        skiprows=3,
        skipfooter=1
    )

    lg.debug("Loaded dataframe:")
    lg.debug(data)

    if cfg_ds.features_to_drop:
        data.drop(cfg_ds.features_to_drop, axis=1, inplace=True)

    data = process_num_df_to_binaryclass(
        data=data,
        label_column=cfg_ds.label_column,
        label_threshold=cfg_ds.label_threshold,
        low_num_feature_val_thr=cfg_ds.low_num_feature_val_thr,
        low_num_feature_samples_thr=cfg_ds.low_num_feature_samples_thr,
    )

    lg.debug("Processed dataframe:")
    lg.debug(data)

    # h2o.init(
    #     nthreads=8,
    #     max_mem_size='12G',
    # )

    lg.info("fin.")


if __name__ == '__main__':
    main()