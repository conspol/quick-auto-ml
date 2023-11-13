from typing import Any, List, Union

from hydra.core.config_store import ConfigStore
from loguru import logger as lg
from omegaconf import DictConfig, ListConfig, OmegaConf

from quick_auto_ml.conf_schema.structured_configs import (
    AppConfig,
    DataConfig,
    InputFileConfig,
)
from quick_auto_ml.defines import (
    CFGOPT_FILE_SAME,
    CFGOPT_FILE_SHEET_SAME,
    ROOT_CONFIG_NAME,
)

_DictType = Union[dict, DictConfig]
_ListType = Union[list, ListConfig]


def register_base_config():
    """
    A mandatory function for Hydra to use the Structured Config schema.
    """
    config_store = ConfigStore.instance()
    config_store.store(name='base_config', node=AppConfig)


def process_random_seeds(config: AppConfig):
    root_seed = config.get('random_seed', None)
    if root_seed is None:
        return config

    _propagate_value_in_nested_dict_dfs(
        config,
        'random_seed',
        root_seed,
    )


def _add_list_items_to_visit(
    list_w_items: _ListType,
    visit_list: List,
    current_k: Any,
) -> None:
    for list_i_, list_v_ in enumerate(list_w_items):
        if isinstance(list_v_, _DictType):
            visit_list.append((f'{current_k}[{list_i_}]', list_v_))


def _propagate_value_in_nested_dict_dfs(
    nested_dict: _DictType,
    key: str,
    value: Any,
) -> None:

    # The first item in this tuple will be displayed in the log message.
    stack = [(ROOT_CONFIG_NAME + ' (root)', nested_dict)]
    changed_dicts = []
    unchanged_dicts = []

    while stack:
        current_lvl_k, current_lvl_v = stack.pop()
        visit_list = list(current_lvl_v.items())

        while len(visit_list) > 0:
            k_, v_ = visit_list.pop()

            if isinstance(v_, _DictType):
                stack.append((k_, v_))
                continue

            elif isinstance(v_, _ListType):
                _add_list_items_to_visit(v_, visit_list, k_)
            
            elif k_ == key and v_ != -1:
                current_lvl_v[k_] = value
                changed_dicts.append(current_lvl_k)

            elif v_ == -1:
                current_lvl_v[k_] = None
                unchanged_dicts.append(current_lvl_k)

    lg.info(f"Changed `{key}` to {value} in: "
            f"{changed_dicts}; left random in: {unchanged_dicts}")


def process_same_file_cfgs(config: DataConfig) -> None:
    for merge_cfg in config.merge_data:
        _process_same_file_cfg(merge_cfg, config)

    if config.test_data is not None:
        for k_, cfg_ in config.test_data.items():
            cfg_obj_ = OmegaConf.to_object(cfg_) if cfg_ else None
            if isinstance(cfg_obj_, InputFileConfig):
                _process_same_file_cfg(cfg_, config)


def _process_same_file_cfg(
    child_cfg: InputFileConfig,
    parent_cfg: InputFileConfig,
) -> None:
    if child_cfg.input_file == CFGOPT_FILE_SAME:
        child_cfg.input_file = parent_cfg.input_file
        child_cfg.input_file_type = parent_cfg.input_file_type

        if (child_cfg.input_file_sheet_name == CFGOPT_FILE_SHEET_SAME
            or child_cfg.input_file_sheet_name is None):
            child_cfg.input_file_sheet_name = parent_cfg.input_file_sheet_name
            child_cfg.skiprows = parent_cfg.skiprows
            child_cfg.skipfooter = parent_cfg.skipfooter

    elif child_cfg.input_file_sheet_name == CFGOPT_FILE_SHEET_SAME:
        raise ValueError(
            "The `input_file_sheet_name` cannot be set to "
            f"`{CFGOPT_FILE_SHEET_SAME}` if the `input_file` is not set to "
            f"`{CFGOPT_FILE_SAME}`. "
        )
