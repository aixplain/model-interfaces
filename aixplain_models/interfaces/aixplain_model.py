from kserve.model import Model
from typing import Dict, List
from aixplain_models.schemas.function_input import APIInput
from aixplain_models.schemas.function_output import APIOutput

class AixplainModel(Model):

    def run_model(self, api_input: Dict[str, List[APIInput]], headers: Dict[str, str] = None) -> Dict[str, List[APIOutput]]:
        pass

    def predict(self, request: Dict[str, List[APIInput]], headers: Dict[str, str] = None) -> Dict[str, List[APIOutput]]:
        pass