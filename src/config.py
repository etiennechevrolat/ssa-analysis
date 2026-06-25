
from pathlib import Path
import yaml
from pydantic import BaseModel

class Constellation(BaseModel):
    name_pattern: str 
    country: str

class DataConfig(BaseModel):
    constellations: list[Constellation]

def load_data_config(path: str = "configs/data.yaml"):
    raw=yaml.safe_load(Path(path).read_text())
    return DataConfig(**raw)
