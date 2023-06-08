## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com
## Example command to run this test:
## ASSET_URI=./docs/user/samples/metric/dnsmos/src/external/weights/  python -m pytest docs/user/samples/metric/dnsmos

import json
import pytest

from docs.user.samples.metric.dnsmos.src.model import DNSMOS
from pathlib import Path

INPUTS_PATH = Path("docs/user/samples/metric/dnsmos/tests/inputs.json")
OUTPUTS_PATH = Path("docs/user/samples/metric/dnsmos/tests/outputs.json")


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
    scorer = DNSMOS("mock-metric")
    scorer.load()
    return scorer


@pytest.mark.parametrize("testcase", ["referenceless1", "referenceless2"])
def test_dnsmos(inputs, outputs, testcase, scorer):
    inp = inputs[testcase]

    inp = {
        "sources": inp["sources"],
        "hypotheses": inp["hypotheses"],
        "supplier": "aiXplain",
        "metric": "dnsmos",
    }
    request = {"instances": [inp]}
    result = scorer.score(request)
    pred_out = result["scores"][0]["details"]

    real_out = outputs[testcase]
    assert real_out == pred_out


def test_dnsmos_aggregation(inputs, scorer):
    metric = "dnsmos"
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
        "data": [{"score": 3.07}],
        "aggregation_metadata": [{"corpus-sum": 16.85, "hyp-length-sum": 5.48}],
        "supplier": "aiXplain",
        "metric": metric,
        "version": "",
    }
