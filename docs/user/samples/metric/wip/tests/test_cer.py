__author__='thiagocastroferreira'

import json
import pytest
import tempfile

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
    with tempfile.NamedTemporaryFile() as hypotheses_path, tempfile.NamedTemporaryFile() as references_path:
        json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(inp['references'], open(references_path.name, 'w'))

        inp = { 
            "hypotheses": hypotheses_path.name,
            "references": references_path.name,
            "supplier": "aiXplain",
            "metric": "cer"
        }

        request = {
            'instances': [inp]
        }
        result = scorer.score(request)
        pred_out = result['scores'][0]['details']

        real_out = outputs[testcase]
        assert real_out == pred_out