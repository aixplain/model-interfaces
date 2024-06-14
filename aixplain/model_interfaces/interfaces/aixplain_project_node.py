__author__='aiXplain'

from abc import abstractmethod
from kserve.model import Model
from typing import Dict

class AixplainProjectNode(Model):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.ready = False
        ready = self.load()
        assert ready

    def predict(self, request: Dict[str, Dict], headers: Dict[str, str] = None) -> Dict[str, Dict]:
        return self.run_script(request, headers)

    @abstractmethod
    def run_script(self, input: Dict[str, Dict], headers: Dict[str, str] = None) -> Dict[str, Dict]:
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