from pydantic import BaseModel
from typing import Dict, Any


class Chart(BaseModel):
    config_template_path: str
    values: Dict[str, Any]
