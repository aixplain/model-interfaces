__author__='thiagocastroferreira'

import json
import pytest
import tempfile

from docs.user.samples.metric.sklearn_classification.src.model import SklearnClassificationMetrics

INPUTS_PATH="docs/user/samples/metric/sklearn_classification/tests/inputs.json"
OUTPUTS_PATH="docs/user/samples/metric/sklearn_classification/tests/outputs.json"

@pytest.fixture
def inputs():
    with open(INPUTS_PATH) as f:
        return json.load(f)


@pytest.fixture
def outputs():
    with open(OUTPUTS_PATH) as f:
        return json.load(f)

@pytest.mark.parametrize("metric", [
    "accuracy", 
    "precision", 
    "recall", 
    "f1", 
    "confusion_matrix", 
    "popular_classification_metrics", 
    "hamming_loss"])
def test_classification(inputs, outputs, metric):
    scorer = SklearnClassificationMetrics()
    
    with tempfile.NamedTemporaryFile() as hypotheses_path, tempfile.NamedTemporaryFile() as references_path:
        json.dump(inputs['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(inputs['references'], open(references_path.name, 'w'))

        inp = { 
            "hypotheses": hypotheses_path.name,
            "references": references_path.name,
            "supplier": "scikit-learn",
            "metric": metric
        }

        request = {
            'instances': [inp]
        }
        result = scorer.score(request)
        pred_out = result['scores'][0]['details']

        real_out = outputs[metric]
        assert real_out == pred_out