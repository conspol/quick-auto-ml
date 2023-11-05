from typing import Any, List, Union

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


