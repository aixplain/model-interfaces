__author__='aiXplain'

import json
import pytest
import tempfile
import os
from docs.user.samples.metric.comet_da.src.model import CometDAMetric

MODEL_PATH = r"C:\Users\shreyas\aiXplain\internal\aixplain-models-internal-assets\wmt-large-da-estimator-1719\model\_ckpt_epoch_1.ckpt"
INPUTS_PATH="docs/user/samples/metric/comet_da/tests/inputs.json"
OUTPUTS_PATH="docs/user/samples/metric/comet_da/tests/outputs.json"

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
    scorer = CometDAMetric()
    scorer.load()
    return scorer


@pytest.mark.parametrize("testcase", ["regular", "multireference", "multireference2", "emptystring"])
def test_comet_da(inputs, outputs, testcase, scorer):
    inp = inputs[testcase]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sources_path, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as references_path:
        json.dump(inp['sources'], open(sources_path.name, 'w'))
        json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(inp['references'], open(references_path.name, 'w'))
        files_to_be_deleted = [sources_path.name, hypotheses_path.name, references_path.name]

        inp = { 
            "sources": sources_path.name,
            "hypotheses": hypotheses_path.name,
            "references": references_path.name,
            "supplier": "unbabel",
            "metric": "comet-da"
        }

        request = {
            'instances': [inp]
        }
        result = scorer.score(request)
        pred_out = result['scores'][0]['details']
    for filename in files_to_be_deleted:
        if os.path.exists(filename):
            os.remove(filename)

    real_out = outputs[testcase]
    assert real_out == pred_out


def test_comet_da_aggregation(inputs, outputs, scorer):
    metric = 'comet-da'
    results = []
    testcases = ["multireference", "multireference2", "emptystring"]
    for testcase in testcases:
        inp = inputs[testcase]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sources_path, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as references_path:
            json.dump(inp['sources'], open(sources_path.name, 'w'))
            json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
            json.dump(inp['references'], open(references_path.name, 'w'))
            files_to_be_deleted = [sources_path.name, hypotheses_path.name, references_path.name]

            inp = { 
                "sources": sources_path.name,
                "hypotheses": hypotheses_path.name,
                "references": references_path.name,
                "supplier": "unbabel",
                "metric": "comet-da"
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
        "instances": [[r['metric_aggregate'] for r in results]]
    }
    result = scorer.aggregate(request)
    result_from_aggregate_function = result['aggregates'][0]['data']
    combined_inp = {'hypotheses': [], 'references': [], 'sources': []}
    for testcase in testcases:
        inp = inputs[testcase]
        combined_inp['sources'] += inp['sources']
        combined_inp['hypotheses'] += inp['hypotheses']
        combined_inp['references'] += inp['references']
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sources_path, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as references_path:
        json.dump(combined_inp['sources'], open(sources_path.name, 'w'))
        json.dump(combined_inp['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(combined_inp['references'], open(references_path.name, 'w'))
        files_to_be_deleted = [sources_path.name, hypotheses_path.name, references_path.name]

        inp = { 
            "sources": sources_path.name,
            "hypotheses": hypotheses_path.name,
            "references": references_path.name,
            "supplier": "unbabel",
            "metric": "comet-da"
        }

        request = {
            'instances': [inp]
        }
        result = scorer.score(request)
        result_from_combined_input = result['scores'][0]['data']
    for filename in files_to_be_deleted:
            if os.path.exists(filename):
                os.remove(filename)

    assert result_from_combined_input == result_from_aggregate_function