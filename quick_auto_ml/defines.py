from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

CLASS_LABEL = 'class_label'

ROOT_CONFIG_NAME = 'AppConfig'

CFGOPT_FILE_SAME = CFGOPT_FILE_SHEET_SAME = 'same'
