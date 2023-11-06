from dataclasses import dataclass, field
from typing import List, Optional, Union
from ..defines import LogLevel

from omegaconf import MISSING


@dataclass
class InputFileConfig:
    input_file: str = MISSING
    input_file_type: str = 'excel'
    input_file_sheet_name: str = MISSING
    index_column: int = 0
    skiprows: int = 0
    skipfooter: int = 0

@dataclass
class MergeWithConfig(InputFileConfig):
    features_to_add: List[str] = field(default_factory=list)
    how: str = 'left'

@dataclass
class TestDataSplitConfig:
    test_size: float = 0.2  # Portion of the data to be used for testing
    random_seed: Optional[int] = None
    stratify_by: Optional[str] = None

@dataclass
class TestDataConfig:
    split: Optional[TestDataSplitConfig] = None
    file: Optional[InputFileConfig] = None
    index_file: Optional[InputFileConfig] = None

@dataclass
class DataConfig(InputFileConfig):
    label_column: str = MISSING
    label_threshold: Union[int, float] = MISSING

    low_num_feature_val_thr: Union[int, float, None] = None
    low_num_feature_samples_thr: Optional[int] = None
    sensitivity_thr: Optional[Union[int, float]] = None
    features_to_drop: Optional[List[str]] = None

    merge_data: List[MergeWithConfig] = field(default_factory=list)
    drop_null_label_samples: bool = True

    test_data: Optional[TestDataConfig] = None

@dataclass
class TrainConfig:
    exclude_cols: Optional[List[str]] = None

@dataclass
class AppConfig:
    log_level: LogLevel = LogLevel.INFO
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    random_seed: Optional[int] = None
    change_nested_random_seeds: bool = True

