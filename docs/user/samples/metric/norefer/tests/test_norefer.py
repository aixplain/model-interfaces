__author__ = "aiXplain"

import json
import pytest
import tempfile

from docs.user.samples.metric.norefer.src.model import NoRefERMetric

INPUTS_PATH = "docs/user/samples/metric/norefer/tests/inputs.json"
OUTPUTS_PATH = "docs/user/samples/metric/norefer/tests/outputs.json"


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
    scorer = NoRefERMetric()
    scorer.load()
    return scorer


@pytest.mark.parametrize("testcase", ["regular", "multireference", "multireference2", "emptystring"])
def test_norefer(inputs, outputs, testcase, scorer):
    inp = inputs[testcase]

    with tempfile.NamedTemporaryFile() as sources_path, tempfile.NamedTemporaryFile() as hypotheses_path:
        json.dump(inp["sources"], open(sources_path.name, "w"))
        json.dump(inp["hypotheses"], open(hypotheses_path.name, "w"))

        inp = {"sources": sources_path.name, "hypotheses": hypotheses_path.name, "supplier": "aiXplain", "metric": "rer"}

        request = {"instances": [inp]}
        result = scorer.score(request)
        pred_out = result["scores"][0]["details"]

        real_out = outputs[testcase]
        assert real_out == pred_out
