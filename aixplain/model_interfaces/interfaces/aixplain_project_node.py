__author__='aiXplain'

from abc import abstractmethod
from kserve.model import Model
from typing import Dict, List
from pydantic import validate_call

class AixplainProjectNode(Model):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name)
        self.name = name
        self.ready = False
        ready = self.load()
        print(f"Readiness: {ready}")
        assert ready

    @validate_call
    def predict(self, request: Dict[str, Dict], headers: Dict[str, str] = None) -> Dict[str, List]:
        print("Calling predict")
        return self.run_script(request)

    @validate_call
    def run_script(self, input: Dict[str, Dict]) -> Dict[str, List]:
        raise NotImplementedError

    def load(self) -> bool:
        """Load handler can be overridden to load the metric from storage
        ``self.ready`` flag is used for metric health check
        :return:
            True if metric is ready, False otherwise
        :rtype:
            Bool
        """
        self.ready = True
        return self.ready