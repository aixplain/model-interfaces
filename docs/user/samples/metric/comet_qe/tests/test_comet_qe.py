__author__='thiagocastroferreira'

import json
import pytest
import tempfile
import os
from docs.user.samples.metric.comet_qe.src.model import CometQEMetric

INPUTS_PATH="docs/user/samples/metric/comet_qe/tests/inputs.json"
OUTPUTS_PATH="docs/user/samples/metric/comet_qe/tests/outputs.json"

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
    scorer = CometQEMetric()
    scorer.load()
    return scorer


@pytest.mark.parametrize("testcase", ["regular", "multireference", "multireference2", "emptystring"])
def test_comet_qe(inputs, outputs, testcase, scorer):
    inp = inputs[testcase]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sources_path, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path:
        json.dump(inp['sources'], open(sources_path.name, 'w'))
        json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
        files_to_be_deleted = [sources_path.name, hypotheses_path.name]

        inp = { 
            "sources": sources_path.name,
            "hypotheses": hypotheses_path.name,
            "supplier": "unbabel",
            "metric": "comet-qe"
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


def test_comet_qe_aggregation(inputs, outputs, scorer):
    metric = 'comet-qe'
    results = []
    testcases = ["multireference", "multireference2", "emptystring"]
    for testcase in testcases:
        inp = inputs[testcase]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sources_path, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path:
            json.dump(inp['sources'], open(sources_path.name, 'w'))
            json.dump(inp['hypotheses'], open(hypotheses_path.name, 'w'))
            files_to_be_deleted = [sources_path.name, hypotheses_path.name]

            inp = { 
                "sources": sources_path.name,
                "hypotheses": hypotheses_path.name,
                "supplier": "unbabel",
                "metric": metric
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
        combined_inp['sources'] += inp['sources']
        combined_inp['hypotheses'] += inp['hypotheses']
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as sources_path, \
         tempfile.NamedTemporaryFile(suffix=".json", delete=False) as hypotheses_path:
        json.dump(combined_inp['sources'], open(sources_path.name, 'w'))
        json.dump(combined_inp['hypotheses'], open(hypotheses_path.name, 'w'))
        files_to_be_deleted = [sources_path.name, hypotheses_path.name]

        inp = { 
            "sources": sources_path.name,
            "hypotheses": hypotheses_path.name,
            "supplier": "unbabel",
            "metric": metric
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