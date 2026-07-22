
from pathlib import Path
import yaml
from pydantic import BaseModel
from datetime import date
from typing import Optional


class Constellation(BaseModel):
    name_pattern: str = ""
    country: str = ""
    norad_ids: Optional[list[int]] = None


class DataRange(BaseModel):
    start: date
    end: date

class OrbitRange(BaseModel):
    borneinf : float
    bornesup : float

class ManeuverConfig(BaseModel):
    sat_name: str

class DataConfig(BaseModel):
    constellations: list[Constellation]
    data_range: DataRange
    orbit_range: Optional[OrbitRange] = None
    maneuvers: Optional[ManeuverConfig] = None

from dataclasses import dataclass

@dataclass
class Config:
    ## Les données
    pass
def load_data_config(path: str = "configs/data.yaml"):
    raw=yaml.safe_load(Path(path).read_text())
    return DataConfig(**raw)
