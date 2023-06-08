## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com

import os
import json
import pytest
import tempfile

from docs.user.samples.metric.nisqa.src.model import Nisqa
from pathlib import Path

INPUTS_PATH = Path("docs/user/samples/metric/nisqa/tests/inputs.json")
OUTPUTS_PATH = Path("docs/user/samples/metric/nisqa/tests/outputs.json")


@pytest.fixture
def inputs():
    with open(INPUTS_PATH) as f:
        return json.load(f)


@pytest.fixture
def outputs():
    with open(OUTPUTS_PATH) as f:
        return json.load(f)


@pytest.fixture
def scorer():
    scorer = Nisqa("mock-metric")
    scorer.load()
    return scorer


@pytest.mark.parametrize("testcase", ["referenceless1", "referenceless2"])
def test_nisqa(inputs, outputs, testcase, scorer):
    inp = inputs[testcase]

    inp = {
        "sources": inp["sources"],
        "hypotheses": inp["hypotheses"],
        "supplier": "aiXplain",
        "metric": "nisqa",
    }
    request = {"instances": [inp]}
    result = scorer.score(request)
    pred_out = result["scores"][0]["details"]

    real_out = outputs[testcase]
    assert real_out == pred_out


def test_nisqa_aggregation(inputs, scorer):
    metric = "nisqa"
    results = []
    testcases = ["referenceless1", "referenceless2"]
    for testcase in testcases:
        inp = inputs[testcase]

        inp = {
            "sources": inp["sources"],
            "hypotheses": inp["hypotheses"],
            "supplier": "aiXplain",
            "metric": metric,
        }
        request = {"instances": [inp]}
        result = scorer.score(request)
        results.append(result["scores"][0])

    request = {"instances": [[r["metric_aggregate"] for r in results]]}
    result = scorer.run_aggregation(request)
    assert result["aggregates"][0].dict() == {
        "data": [{"score": 3.43}],
        "aggregation_metadata": [{"corpus-sum": 18.79, "hyp-length-sum": 5.48}],
        "supplier": "aiXplain",
        "metric": metric,
        "version": "",
    }
