from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger as lg
from sklearn.model_selection import train_test_split

from quick_auto_ml.conf_schema.structured_configs import (
    DataConfig,
    InputFileConfig,
    MergeWithConfig,
)
from quick_auto_ml.defines import CLASS_LABEL


def load_dataframe(cfg_file: InputFileConfig) -> pd.DataFrame:
    """
    Loads the data from the input file.
    **Only `.xlsx` files are supported for now.**

    Parameters
    ----------
    cfg_file : InputFileConfig
        The input file configuration.
    """
    data = pd.read_excel(
        cfg_file.input_file,
        index_col=cfg_file.index_column,
        sheet_name=cfg_file.input_file_sheet_name,
        skiprows=cfg_file.skiprows,
        skipfooter=cfg_file.skipfooter,
    )
    data.columns = data.columns.str.strip()

    return data


def get_num_features_data(
    data: pd.DataFrame,
    label_column: Optional[str] = None,
    preserve_label: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with numeric features only.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe.
    label_column : str
        Name of the column containing labels (to be preserved).
    """
    exclude_from_filter = []
    if label_column is not None:
        if preserve_label:
            exclude_from_filter = [label_column]
        else:
            lg.warning(
                "Label column specified, but `preserve_label` is False; "
                "label column will be filtered out."
            )
    elif preserve_label:
        raise ValueError(
            "If `preserve_label` is True, `label_column` must be specified."
        )

    num_data = data.select_dtypes(include=[np.number])
    num_data = num_data[num_data.columns.difference([*exclude_from_filter])]
    return num_data


def process_num_df_to_binaryclass(
    data: pd.DataFrame,
    label_column: str,
    label_threshold: Union[float, int],
    low_num_feature_val_thr: Union[float, int] = None,
    low_num_feature_samples_thr: int = None,
    sensitivity_thr: Union[float, int] = None,
    drop_null_label_samples: bool = True,
) -> pd.DataFrame:
    """
    Processes a DataFrame: converts numeric labels into binary class labels 
    and filters out features with values below `low_num_feature_val_thr` in more 
    than `low_num_feature_samples_thr` samples.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe.
    label_column : str
        Name of the column containing numeric labels to be converted.
    label_threshold : Union[float, int]
        Threshold above which labels are converted to 'P', otherwise 'N'.
    low_num_feature_val_thr : Union[float, int], optional
        Value below which a feature is considered low.
    low_num_feature_samples_thr : int, optional
        Minimum number of samples with low value for feature to be considered
        for removal.
    sensitivity_thr : Union[float, int], optional
        Threshold below which a feature is considered zero.
    drop_null_label_samples : bool, default True
        If True, drops samples with null values in the label column.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with binary class labels and filtered features.

    Notes
    -----
    If both `low_num_feature_val_thr` and `low_num_feature_samples_thr` are not 
    specified, the feature filtration based on low values is skipped.
    """
    if drop_null_label_samples:
        data=data[~(data[label_column].isnull())]

    data[CLASS_LABEL]='N'
    data.loc[data[label_column] > label_threshold, CLASS_LABEL] = 'P'

    num_data = None

    if low_num_feature_val_thr or low_num_feature_samples_thr:
        if low_num_feature_val_thr and low_num_feature_samples_thr:
            num_data = get_num_features_data(
                data=data,
                label_column=label_column,
            )

            low_vals_samples = (
                num_data < low_num_feature_val_thr
            ).sum(axis=0)

            filtered_features = low_vals_samples[
                low_vals_samples > low_num_feature_samples_thr
            ].index.to_list()

            data.drop(filtered_features, axis=1, inplace=True)
            lg.info(
                f"Filtered features with too many low-valued samples: "
                f"{filtered_features}"
            )

        else:
            lg.warning(
                "Both *low_num_feature_val_thr* and "
                "*low_num_feature_samples_thr* must be specified; "
                "skipping filtration."
            )

    if sensitivity_thr:
        if num_data is None:
            num_data = get_num_features_data(
                data=data,
                label_column=label_column,
            )

        zero_vals_samples = num_data < sensitivity_thr

        data[zero_vals_samples] = 0

    return data


def merge_data(
    data: pd.DataFrame,
    merge_cfg: MergeWithConfig,
) -> pd.DataFrame:
    """
    Merges the data from another file.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe.
    merge_cfg : MergeWithConfig
        The configuration for merging the data.
    """
    data2merge = load_dataframe(cfg_file=merge_cfg)

    data2merge = data2merge[merge_cfg.features_to_add]
    data = data.join(data2merge, how=merge_cfg.how)

    lg.info(
        f"Merged columns: {data2merge.columns.to_list()}\n"
        f"\tfrom file {merge_cfg.input_file}\n"
        f"\tfrom sheet {merge_cfg.input_file_sheet_name}"
    )

    return data


def prepare_data(
    data: pd.DataFrame,
    cfg_ds: DataConfig,
) -> pd.DataFrame:
    """
    Prepares the data for training: drops features, converts numeric labels,
    merges data from other files.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe.
    cfg_ds : DictConfig
        The dataset configuration.
    """
    if cfg_ds.features_to_drop:
        data.drop(cfg_ds.features_to_drop, axis=1, inplace=True)

    data = process_num_df_to_binaryclass(
        data=data,
        label_column=cfg_ds.label_column,
        label_threshold=cfg_ds.label_threshold,
        low_num_feature_val_thr=cfg_ds.low_num_feature_val_thr,
        low_num_feature_samples_thr=cfg_ds.low_num_feature_samples_thr,
        sensitivity_thr=cfg_ds.sensitivity_thr,
        drop_null_label_samples=cfg_ds.drop_null_label_samples,
    )

    if cfg_ds.merge_data:
        for merge_cfg in cfg_ds.merge_data:
            data = merge_data(data=data, merge_cfg=merge_cfg)

    return data


def prepare_test_train(
    data: pd.DataFrame,
    cfg_ds: DataConfig,
) -> (pd.DataFrame, Union[pd.DataFrame, None]):
    """
    Conducts the train-test split of a dataframe as per the `cfg_ds` configurations.

    The function allows for three distinct methods of splitting, mutually exclusive:
    - `split`: Parameters defined for `train_test_split` function of sklearn.
    - `index_file`: A file with indices designating the test dataset.
    - `file`: A direct file input for the test data - *not implemented yet*.

    If no `test_data` configurations are given, defaults to utilizing all data 
    as the training set.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe to be split.
    cfg_ds : DataConfig
        An object that contains the configuration for the test data, including:
        - `split`: The proportion and parameters for the split.
        - `index_file`: The path and specifications of the file containing 
            test indices.
        - `file`: The path and specifications for the file to load test data from
            (*Not Implemented yet*).

    Returns
    -------
    data_train : pd.DataFrame
        The training set derived from the input data.
    data_test : pd.DataFrame or None
        The testing set derived from the input data or None if no test data
        configuration is specified.

    Raises
    ------
    ValueError
        If multiple test_data configurations are provided or if no proper
        configuration is specified.
    """
    if cfg_ds.test_data:
        config_options = [
            cfg_ds.test_data.file,
            cfg_ds.test_data.index_file,
            cfg_ds.test_data.split,
        ]
        config_count = sum(1 for option in config_options if option)

        if config_count > 1:
            raise ValueError("Multiple `test_data` configurations provided. "
                             "Please specify only one.")

        # TODO: test data preparation from file
        if cfg_ds.test_data.file:
            raise NotImplementedError(
                "Loading test data from file is not fully implemented yet."
            )
            data_test = load_dataframe(cfg_file=cfg_ds.test_data.file)
            data_train = data
            lg.info(
                f"Loaded test data from file: "
                f"{cfg_ds.test_data.file.input_file}"
            )

        elif cfg_ds.test_data.index_file:
            test_indices = load_dataframe(cfg_file=cfg_ds.test_data.index_file)
            test_indices = test_indices.index
            data_train = data.loc[~data.index.isin(test_indices)]
            data_test = data.loc[data.index.isin(test_indices)]
            lg.info(
                f"Train-test split according to indices taken from file: "
                f"{cfg_ds.test_data.index_file.input_file}\n"
                f"from sheet: {cfg_ds.test_data.index_file.input_file_sheet_name}"
            )
            lg.info(
                f"Train size: {len(data_train)}; test size: {len(data_test)}"
            )

            lg.debug(f"Test indices: {data_test.index.to_list()}")

        elif cfg_ds.test_data.split:
            data_train, data_test = train_test_split(
                data,
                test_size=cfg_ds.test_data.split.test_size,
                random_state=cfg_ds.test_data.split.random_seed,
                stratify=data[cfg_ds.test_data.split.stratify_by],
            )
        else:
            raise ValueError("No proper `test_data` configuration specified.")

    else:
        data_train = data
        data_test = None
        lg.info("No test data specified; using all data for training.")

    return data_train, data_test