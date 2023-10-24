from dataclasses import dataclass, field
from typing import List, Optional, Union
from ..defines import LogLevel

from omegaconf import MISSING


@dataclass
class DataConfig:
    input_file: str = MISSING
    input_file_type: str = 'xlsx'
    input_file_sheet_name: str = MISSING
    label_column: str = MISSING
    label_threshold: Union[int, float] = MISSING

    low_num_feature_val_thr: Union[int, float, None] = None
    low_num_feature_samples_thr: Union[int, None] = None
    features_to_drop: Optional[List[str]] = None

@dataclass
class AppConfig:
    random_seed: int = 42
    log_level: LogLevel = LogLevel.INFO
    data: DataConfig = field(default_factory=DataConfig)