__author__='aiXplain'

import numpy as np
import os

from aixplain.model_schemas.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_schemas.interfaces.metric_models import TextGenerationMetric
from aixplain.model_schemas.schemas.metric_input import TextGenerationMetricInput, MetricAggregate
from aixplain.model_schemas.interfaces.asset_resolver import AssetResolver
from aixplain.model_schemas.schemas.metric_output import TextGenerationMetricOutput
from comet import download_model, load_from_checkpoint
import aixplain.model_schemas.utils.metric_utils as metric_utils
import itertools
from typing import Dict, List, Union

MODEL_NOT_FOUND_ERROR = """
    Download model file using command:
    # TODO (krishnadurai): Host this on a public URL
    aws s3 cp --recursive https://benchmarksdata.s3.amazonaws.com/models/wmt-large-da-estimator-1719/_ckpt_epoch_1.ckpt .
"""

class CometDAMetric(TextGenerationMetric):
    params = {
        "casing" : ["Cased", "Uncased"],
        "punctuation" : ["Punctuated", "Not Punctuated"]
    }

    def __init__(self, name="comet-da", *args, **kwargs):
        self.all_possible_configs = [dict(zip(self.params, vals)) for vals in itertools.product(*self.params.values())]
        super().__init__(name, *args, **kwargs)

    def load(self, model_name:str="wmt21-comet-da", batch_size:int=16, gpus:int=0):
        model_path = AssetResolver.resolve_path()
        model_path = download_model(model_name)
        # model_path = model_name
        if not os.path.exists(model_path):
            raise ValueError(MODEL_NOT_FOUND_ERROR)
        self.model = load_from_checkpoint(model_path)
        self.gpus = gpus
        self.batch_size = batch_size
        self.ready = True

    def run_aggregation(self, request: Dict[str, List[List[MetricAggregate]]]) -> Dict[str, List[MetricAggregate]]:
        """Aggregation function to aggregate previous computed scores

        Args:
            request (Dict[str, List[List[MetricAggregate]]]): outputs of the APIs

        Returns:
            Dict[str, List[MetricAggregate]]: _description_
        """
        predictions = []
        batches = request["instances"]
        for batch_info in batches:
            aggregate_metadata_list, data_list = [], []
            num_sub_samples = len(batch_info[0].aggregation_metadata)
            for sub_sample_index in range(num_sub_samples):
                scores, weights = [], []
                for sample_info in batch_info:
                    sub_sample_info = sample_info.aggregation_metadata[sub_sample_index]
                    scores.append(sub_sample_info['corpus-level-score'])
                    weights.append(sub_sample_info['hypotheses_count'])
                aggregate_score = round(np.average(scores, weights=weights), 4)
                aggregate_metadata = {
                    "corpus-level-score" : aggregate_score,
                    "hypotheses_count" : np.sum(weights),
                    "options" : sub_sample_info['options']
                }
                data = {
                    "score" : round(aggregate_score, 4),
                    "options" : sub_sample_info['options']
                }
                data_list.append(data)
                aggregate_metadata_list.append(aggregate_metadata)
            output_dict = MetricAggregate(**{
                "data": data_list,
                "aggregation_metadata": aggregate_metadata_list,
                "supplier" : batch_info[0].supplier,
                "metric" : batch_info[0].metric,
                "version" : batch_info[0].version 
            })
            predictions.append(output_dict)
        predict_output = {"aggregates": predictions}
        return predict_output

        
    def run_metric(self, request: Dict[str, List[TextGenerationMetricInput]]) -> Dict[str, List[TextGenerationMetricOutput]]:
        """Scoring Function for Unbabel Comet Metrics

        Args:
            request (Dict[str, List[TextGenerationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[TextGenerationMetricOutput]]: outuput
        """
        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            data_list, details_list, aggregate_metadata_list = [], [], []
            for current_config in self.all_possible_configs:
                details_dict = {}
                hypotheses, references, sources = inp.hypotheses.copy(), inp.references.copy(), inp.sources.copy()
                if current_config["casing"] == "Uncased":
                    hypotheses = metric_utils.lowercase(hypotheses)
                    sources = metric_utils.lowercase(sources)
                    for j, reference in enumerate(references):
                        references[j] = metric_utils.lowercase(reference)
                if current_config["punctuation"] == "Not Punctuated":
                    sources = metric_utils.remove_punctuation(sources)
                    hypotheses = metric_utils.remove_punctuation(hypotheses)
                    for j, reference in enumerate(references):
                        references[j] = metric_utils.remove_punctuation(reference)

                seg_scores, nreferences = [], len(references[0])
                for nref in range(nreferences):
                    references_ = [refs[nref] for refs in references]
                    data = [{ "src": sources[i], "mt": hypotheses[i], "ref": references_[i] } for i in range(len(hypotheses))]
                    seg_scores_, _ = self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus)
                    seg_scores.append(seg_scores_)
                seg_scores = np.max(seg_scores, axis=0)
                seg_scores = list(np.round(seg_scores, 4))
                corpus_score = round(np.mean(seg_scores), 4)
                details_dict['sentence-level'] = seg_scores
                details_dict['corpus-level'] =  corpus_score
                details_dict['options'] = current_config
                details_list.append(details_dict)
                data = {
                    "score" : corpus_score,
                    "options" : current_config
                }
                data_list.append(data)
                aggregation_metadata = {
                    'corpus-level-score' : corpus_score,
                    'hypotheses_count' : len(hypotheses),
                    'options' : current_config
                }
                aggregate_metadata_list.append(aggregation_metadata)
            metric_aggregate = MetricAggregate(**{
                "aggregation_metadata": aggregate_metadata_list,
                "supplier" : inp.supplier,
                "metric" : inp.metric,
                "version" : inp.version 
            })
            output_dict = TextGenerationMetricOutput(**{
                "data": data_list,
                "details": details_list,
                "metric_aggregate" : metric_aggregate
            })
            predictions.append(output_dict)
        
        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    metric = CometDAMetric(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([metric])