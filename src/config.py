
from pathlib import Path
import yaml
from pydantic import BaseModel
from datetime import date


class Constellation(BaseModel):
    name_pattern: str 
    country: str


class DataRange(BaseModel):
    start: date
    end: date

class DataConfig(BaseModel):
    constellations: list[Constellation]
    data_range: DataRange


def load_data_config(path: str = "configs/data.yaml"):
    raw=yaml.safe_load(Path(path).read_text())
    return DataConfig(**raw)
