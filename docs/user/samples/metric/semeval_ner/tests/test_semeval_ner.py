__author__='thiagocastroferreira'

import json
import pytest
import tempfile

from docs.user.samples.metric.semeval_ner.src.model import SemEvalNERMetric

INPUTS_PATH="docs/user/samples/metric/semeval_ner/tests/inputs.json"
OUTPUTS_PATH="docs/user/samples/metric/semeval_ner/tests/outputs.json"

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
    scorer = SemEvalNERMetric()
    return scorer


@pytest.mark.parametrize("testcase", ["success", "complex", "complex2"])
def test_semeval_ner(inputs, outputs, testcase, scorer):
    inp = inputs[testcase]

    with tempfile.NamedTemporaryFile() as hypotheses_path, \
         tempfile.NamedTemporaryFile() as references_path:
        json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(inp['references'], open(references_path.name, 'w'))

        inp = { 
            "hypotheses": hypotheses_path.name,
            "references": references_path.name,
            "supplier": "aiXplain",
            "metric": "semeval_ner"
        }

        request = {
            'instances': [inp]
        }
        result = scorer.score(request)
        pred_out = result['scores'][0]['details']

        real_out = outputs[testcase]
        assert real_out == pred_out