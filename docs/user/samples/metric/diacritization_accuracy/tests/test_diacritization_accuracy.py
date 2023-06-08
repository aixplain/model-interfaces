__author__='thiagocastroferreira'

import json
import pytest
import tempfile

from docs.user.samples.metric.diacritization_accuracy.src.model import DiacritizationAccuracy

INPUTS_PATH="docs/user/samples/metric/diacritization_accuracy/tests/inputs.json"
OUTPUTS_PATH="docs/user/samples/metric/diacritization_accuracy/tests/outputs.json"

@pytest.fixture
def inputs():
    with open(INPUTS_PATH) as f:
        return json.load(f)


@pytest.fixture
def outputs():
    with open(OUTPUTS_PATH) as f:
        return json.load(f)

def test_diacritization_accuracy(inputs, outputs):
    scorer = DiacritizationAccuracy()
    metric = "diacritization_accuracy"
    
    with tempfile.NamedTemporaryFile() as hypotheses_path, tempfile.NamedTemporaryFile() as references_path:
        json.dump(inputs['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(inputs['references'], open(references_path.name, 'w'))

        inp = { 
            "hypotheses": hypotheses_path.name,
            "references": references_path.name,
            "supplier": "aiXplain",
            "metric": metric
        }

        request = {
            'instances': [inp]
        }
        result = scorer.score(request)
        pred_out = result['scores'][0]['details']

        real_out = outputs
        assert real_out == pred_out