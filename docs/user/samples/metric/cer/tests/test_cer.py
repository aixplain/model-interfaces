__author__='aiXplain'

import json
import pytest
import tempfile
import os

from docs.user.samples.metric.cer.src.model import CER

INPUTS_PATH="docs/user/samples/metric/cer/tests/inputs.json"
OUTPUTS_PATH="docs/user/samples/metric/cer/tests/outputs.json"

@pytest.fixture
def inputs():
    with open(INPUTS_PATH) as f:
        return json.load(f)


@pytest.fixture
def outputs():
    with open(OUTPUTS_PATH) as f:
        return json.load(f)


@pytest.mark.parametrize("testcase", ["regular", "multireference", "multireference2", "emptystring"])
def test_cer(inputs, outputs, testcase):
    scorer = CER()
    
    inp = inputs[testcase]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path, tempfile.NamedTemporaryFile(suffix=".json", delete=False) as references_path:
        json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(inp['references'], open(references_path.name, 'w'))
        files_to_be_deleted = [hypotheses_path.name, references_path.name]
        inp = { 
            "hypotheses": hypotheses_path.name,
            "references": references_path.name,
            "supplier": "aiXplain",
            "metric": "CER"
        }

        request = {
            'instances': [inp]
        }
        result = scorer.score(request)
        pred_out = result['scores'][0]['details']

        real_out = outputs[testcase]
        assert real_out == pred_out
    for filename in files_to_be_deleted:
        if os.path.exists(filename):
            os.remove(filename)


def test_cer_aggregation(inputs, outputs):
    scorer = CER()
    results = []
    for testcase in ["regular", "multireference", "multireference2", "emptystring"]:
        inp = inputs[testcase]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path, tempfile.NamedTemporaryFile(suffix=".json", delete=False) as references_path:
            json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
            json.dump(inp['references'], open(references_path.name, 'w'))
            files_to_be_deleted = [hypotheses_path.name, references_path.name]

            inp = { 
                "hypotheses": hypotheses_path.name,
                "references": references_path.name,
                "supplier": "aiXplain",
                "metric": "CER"
            }

            request = {
                'instances': [inp]
            }
            result = scorer.score(request)
            results.append(result['scores'][0])
        for filename in files_to_be_deleted:
            if os.path.exists(filename):
                os.remove(filename)


    request = {
        "instances": [results]
    }
    result = scorer.aggregate(request)
    pred_out = result["scores"][0]["data"]
    real_out = outputs["aggregation"]
    assert real_out == pred_out