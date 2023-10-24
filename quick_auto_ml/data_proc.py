from typing import Union, List
import pandas as pd
import numpy as np
from loguru import logger as lg


def process_num_df_to_binaryclass(
    data: pd.DataFrame,
    label_column: str,
    label_threshold: Union[float, int],
    low_num_feature_val_thr: Union[float, int] = None,
    low_num_feature_samples_thr: int = None,
    drop_null_label_samples: bool = True,
) -> pd.DataFrame:
    """
    Processes a DataFrame: converts numeric labels into binary class labels 
    and filters out features with values below `low_num_feature_val_thr` in more 
    than `low_num_feature_samples_thr` samples.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe with numeric data.
    label_column : str
        Name of the column containing numeric labels to be converted.
    label_threshold : Union[float, int]
        Threshold above which labels are converted to 'P', otherwise 'N'.
    low_num_feature_val_thr : Union[float, int], optional
        Value below which a feature is considered low.
    low_num_feature_samples_thr : int, optional
        Minimum number of samples with low value for feature to be considered for removal.
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

    data['class_label']='N'
    data.loc[data[label_column] > label_threshold, 'class_label'] = 'P'

    #Remove a feature if it was too low in 10 or more samples
    if low_num_feature_val_thr is not None \
            or low_num_feature_samples_thr is not None:

        if low_num_feature_val_thr is not None \
                and low_num_feature_samples_thr is not None:

            num_data = data.select_dtypes(include=[np.number])
            low_vals_samples = (
                num_data < low_num_feature_val_thr
            ).sum(axis=0)

            filtered_features = low_vals_samples[
                low_vals_samples > low_num_feature_samples_thr
            ].index.to_list()

            data.drop(filtered_features, axis=1, inplace=True)

        else:
            lg.warning(
                "Both *low_num_feature_val_thr* and "
                "*low_num_feature_samples_thr* must be specified; "
                "skipping filtration."
            )

    return data