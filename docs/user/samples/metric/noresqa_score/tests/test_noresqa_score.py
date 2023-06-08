## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com

import os
import json
import pytest
import tempfile

from docs.user.samples.metric.noresqa_score.src.model import NoresqaScore
from pathlib import Path

INPUTS_PATH = Path("docs/user/samples/metric/noresqa_score/tests/inputs.json")
OUTPUTS_PATH = Path("docs/user/samples/metric/noresqa_score/tests/outputs.json")


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
    scorer = NoresqaScore("mock-metric")
    scorer.load()
    return scorer


@pytest.mark.parametrize("testcase", ["regular", "same"])
def test_noresqa_score(inputs, outputs, testcase, scorer):
    inp = inputs[testcase]

    inp = {
        "sources": inp["sources"],
        "hypotheses": inp["hypotheses"],
        "references": inp["references"],
        "supplier": "aiXplain",
        "metric": "noresqa_score",
    }
    request = {"instances": [inp]}
    result = scorer.score(request)
    pred_out = result["scores"][0]["details"]

    real_out = outputs[testcase]
    assert real_out == pred_out


def test_noresqa_score_aggregation(inputs, scorer):
    metric = "noresqa_score"
    results = []
    testcases = ["regular", "same"]
    for testcase in testcases:
        inp = inputs[testcase]

        inp = {
            "sources": inp["sources"],
            "hypotheses": inp["hypotheses"],
            "references": inp["references"],
            "supplier": "aiXplain",
            "metric": metric,
        }
        request = {"instances": [inp]}
        result = scorer.score(request)
        results.append(result["scores"][0])

    request = {"instances": [[r["metric_aggregate"] for r in results]]}
    result = scorer.run_aggregation(request)
    assert result["aggregates"][0].dict() == {
        "data": [{"score": 7.29}],
        "aggregation_metadata": [{"corpus-sum": 39.93, "hyp-length-sum": 5.48}],
        "supplier": "aiXplain",
        "metric": "noresqa_score",
        "version": "",
    }
