## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com

import json
from pathlib import Path

import pytest

from docs.user.samples.metric.pesq.src.model import PESQ


INPUTS_PATH = Path("docs/user/samples/metric/pesq/tests/inputs.json")
OUTPUTS_PATH = Path("docs/user/samples/metric/pesq/tests/outputs.json")


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
    scorer = PESQ("mock-metric")
    scorer.load()
    return scorer


@pytest.mark.parametrize("testcase", ["regular", "same"])
def test_pesq(inputs, outputs, testcase, scorer):
    inp = inputs[testcase]

    inp = {
        "sources": inp["sources"],
        "hypotheses": inp["hypotheses"],
        "references": inp["references"],
        "supplier": "aiXplain",
        "metric": "pesq",
    }
    request = {"instances": [inp]}
    result = scorer.score(request)
    pred_out = result["scores"][0]["details"]

    real_out = outputs[testcase]
    assert real_out == pred_out


def test_pesq_aggregation(inputs, scorer):
    metric = "pesq"
    results = []
    testcases = ["regular", "same"]
    for testcase in testcases:
        inp = inputs[testcase]

        inp = {
            "sources": inp["sources"],
            "hypotheses": inp["hypotheses"],
            "references": inp["references"],
            "supplier": "aiXplain",
            "metric": "pesq",
        }
        request = {"instances": [inp]}
        result = scorer.score(request)
        results.append(result["scores"][0])

    request = {"instances": [[r["metric_aggregate"] for r in results]]}
    result = scorer.run_aggregation(request)
    assert result["aggregates"][0].dict() == {
        "data": [{"score": 4.03}],
        "aggregation_metadata": [{"corpus-sum": 22.07, "hyp-length-sum": 5.48}],
        "supplier": "aiXplain",
        "metric": "pesq",
        "version": "",
    }
