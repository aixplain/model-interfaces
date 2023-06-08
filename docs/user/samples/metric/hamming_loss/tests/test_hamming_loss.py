__author__='thiagocastroferreira'

import json
import pytest
import tempfile

from docs.user.samples.metric.hamming_loss.src.model import HammingLoss

INPUTS_PATH="docs/user/samples/metric/hamming_loss/tests/inputs.json"
OUTPUTS_PATH="docs/user/samples/metric/hamming_loss/tests/outputs.json"

@pytest.fixture
def inputs():
    with open(INPUTS_PATH) as f:
        return json.load(f)


@pytest.fixture
def outputs():
    with open(OUTPUTS_PATH) as f:
        return json.load(f)


def test_classification(inputs, outputs):
    scorer = HammingLoss()
    
    with tempfile.NamedTemporaryFile() as hypotheses_path, tempfile.NamedTemporaryFile() as references_path:
        json.dump(inputs['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(inputs['references'], open(references_path.name, 'w'))

        inp = { 
            "hypotheses": hypotheses_path.name,
            "references": references_path.name,
            "supplier": "scikit-learn",
            "metric": "hamming_loss"
        }

        request = {
            'instances': [inp]
        }
        result = scorer.score(request)
        pred_out = result['scores'][0]['details']

        real_out = outputs
        assert real_out == pred_out