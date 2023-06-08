## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com

import json
import pytest
import tempfile

from docs.user.samples.metric.visqol.src.model import Visqol
from pathlib import Path

INPUTS_PATH = Path("docs/user/samples/metric/visqol/tests/inputs.json")
OUTPUTS_PATH = Path("docs/user/samples/metric/visqol/tests/outputs.json")


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
    scorer = Visqol("mock-metric")
    scorer.load()
    return scorer


@pytest.mark.parametrize("testcase", ["regular", "same"])
def test_visqol(inputs, outputs, testcase, scorer):
    inp = inputs[testcase]

    inp = {
        "sources": inp["sources"],
        "hypotheses": inp["hypotheses"],
        "references": inp["references"],
        "supplier": "aiXplain",
        "metric": "visqol",
    }
    request = {"instances": [inp]}
    result = scorer.score(request)
    pred_out = result["scores"][0]["details"]

    real_out = outputs[testcase]
    assert real_out == pred_out


def test_visqol_aggregation(inputs, scorer):
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
            "metric": "visqol",
        }
        request = {"instances": [inp]}
        result = scorer.score(request)
        results.append(result["scores"][0])

    request = {"instances": [[r["metric_aggregate"] for r in results]]}
    result = scorer.run_aggregation(request)
    assert result["aggregates"][0].dict() == {
        "data": [{"score": 3.25}],
        "aggregation_metadata": [{"corpus-sum": 17.8, "hyp-length-sum": 5.48}],
        "supplier": "aiXplain",
        "metric": "visqol",
        "version": "",
    }
