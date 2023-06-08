__author__='aiXplain'

import json
import pytest
import tempfile
import os

from docs.user.samples.metric.clsss.src.model import CLSSS

MODEL_PATH= r"C:\Users\shreyas\aiXplain\internal\aixplain-models-internal-assets\LaBSE"
INPUTS_PATH="docs/user/samples/metric/clsss/tests/inputs.json"
OUTPUTS_PATH="docs/user/samples/metric/clsss/tests/outputs.json"

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
    scorer = CLSSS("clsss")
    scorer.load(MODEL_PATH)
    return scorer




@pytest.mark.parametrize("testcase", ["regular", "multireference", "multireference2", "emptystring"])
def test_clsss(inputs, outputs, testcase, scorer):
    
    inp = inputs[testcase]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path, tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sources_path:
        json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(inp['sources'], open(sources_path.name, 'w'))
        files_to_be_deleted = [hypotheses_path.name, sources_path.name]
        inp = { 
            "hypotheses": hypotheses_path.name,
            "sources": sources_path.name,
            "supplier": "aiXplain",
            "metric": "CLSSS"
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


def test_bleu_aggregation(inputs, outputs, scorer):
    metric = 'CLSSS'
    results = []
    testcases = ["regular", "multireference", "multireference2", "emptystring"]
    for testcase in testcases:
        inp = inputs[testcase]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path, tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sources_path:
            json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
            json.dump(inp['sources'], open(sources_path.name, 'w'))
            files_to_be_deleted = [hypotheses_path.name, sources_path.name]
            inp = { 
                "hypotheses": hypotheses_path.name,
                "sources": sources_path.name,
                "supplier": "aiXplain",
                "metric": "CLSSS"
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

    combined_inp = {'hypotheses': [], 'sources': []}
    for testcase in testcases:
        inp = inputs[testcase]
        combined_inp['hypotheses'] += inp['hypotheses']
        combined_inp['sources'] += inp['sources']
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path, tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sources_path:
        json.dump(combined_inp['hypotheses'], open(hypotheses_path.name, 'w'))
        json.dump(combined_inp['sources'], open(sources_path.name, 'w'))
        files_to_be_deleted = [hypotheses_path.name, sources_path.name]
        inp = { 
            "hypotheses": hypotheses_path.name,
            "sources": sources_path.name,
            "supplier": "aiXplain",
            "metric": "CLSSS"
        }

        request = {
            'instances': [inp]
        }
        result = scorer.score(request)
        result_from_combined_input = result['scores'][0]['data']

    assert result_from_combined_input == result_from_aggregate_function