from typing import Union, List
import pandas as pd
import numpy as np
from loguru import logger as lg
from quick_auto_ml.defines import CLASS_LABEL


def _get_num_features_data(
    data: pd.DataFrame,
    label_column: str,
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
    num_data = data.select_dtypes(include=[np.number])
    num_data = num_data[num_data.columns.difference([label_column])]
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
        Minimum number of samples with low value for feature to be considered for removal.
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

    if low_num_feature_val_thr is not None \
            or low_num_feature_samples_thr is not None:

        if low_num_feature_val_thr is not None \
                and low_num_feature_samples_thr is not None:

            num_data = _get_num_features_data(
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
            lg.debug(
                f"Filtered features with too many low-valued samples: "
                f"{filtered_features}"
            )

        else:
            lg.warning(
                "Both *low_num_feature_val_thr* and "
                "*low_num_feature_samples_thr* must be specified; "
                "skipping filtration."
            )

    if sensitivity_thr is not None:
        if num_data is None:
            num_data = _get_num_features_data(
                data=data,
                label_column=label_column,
            )

        zero_vals_samples = num_data < sensitivity_thr

        data[zero_vals_samples] = 0

    return data