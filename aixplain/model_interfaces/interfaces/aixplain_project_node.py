__author__='aiXplain'

from abc import abstractmethod
from enum import Enum
from kserve.model import Model
import inspect
import json
import logging
import time
from typing import Dict, List

from aixplain.model_interfaces.schemas.metric_input import MetricInput, MetricAggregate
from aixplain.model_interfaces.schemas.metric_output import MetricOutput

class AixplainProjectNode(Model):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.ready = False

    def predict(self, request: Dict[str, Dict], headers: Dict[str, str] = None) -> Dict[str, Dict]:
        return self.run_script(request, headers)

    def run_script(self, input: Dict[str, Dict], headers: Dict[str, str] = None) -> Dict[str, Dict]:
        pass

    async def __call__(self, body: Dict,
                       metric_type: MetricType = MetricType.SCORE,
                       headers: Dict[str, str] = None) -> Dict:
        """Method to call scoring or aggregation with the given input.
        :param body:
            Request payload body.
        :type body:
            Dict
        :param metric_type:
            Metric type enum. Can be either scoring or aggregation.
        :type metric_type:
            MetricType
        :param headers:
            Request headers.
        :type headers:
            Dict
        :return:
            Response output from Metric endpoint function - score or aggregate
        :rtype:
            Dict
        """
        request_id = headers.get("x-request-id", "N.A.") if headers else "N.A."

        # latency vars
        aggregate_ms = 0
        score_ms = 0

        body = json.loads(body) #TODO(krishnadurai): Check if this is the best place for JSON conversion
        if metric_type == MetricType.AGGREGATE:
            start = time.time()
            response = (await self.aggregate(body, headers)) if inspect.iscoroutinefunction(self.aggregate) \
                else self.aggregate(body, headers)
            aggregate_ms = get_latency_ms(start, time.time())
        elif metric_type == MetricType.SCORE:
            start = time.time()
            response = (await self.score(body, headers)) if inspect.iscoroutinefunction(self.score) \
                else self.score(body, headers)
            score_ms = get_latency_ms(start, time.time())
        else:
            raise NotImplementedError()

        if self.enable_latency_logging is True:
            logging.info(f"requestId: {request_id},"
                         f"score_ms: {score_ms}, aggregate_ms: {aggregate_ms}, ")

        return response

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