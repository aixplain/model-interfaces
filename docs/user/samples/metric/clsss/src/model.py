__author__='aiXplain'

import numpy as np
import torch

from aixplain.model_interfaces.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_interfaces.interfaces.metric_models import ReferencelessTextGenerationMetric
from aixplain.model_interfaces.schemas.metric_input import ReferencelessTextGenerationMetricInput, MetricAggregate
from aixplain.model_interfaces.interfaces.asset_resolver import AssetResolver
from aixplain.model_interfaces.schemas.metric_output import ReferencelessTextGenerationMetricOutput
from transformers import BertModel, BertTokenizer
from typing import Dict, List
import itertools

MODEL_NOT_FOUND_ERROR = """
    Download model file using command:
    # TODO (krishnadurai): Host this on a public URL
    aws s3 cp --recursive https://benchmarksdata.s3.amazonaws.com/models/LaBSE/ .
"""

class CLSSS(ReferencelessTextGenerationMetric):
    params = {
        "Paramns" : [None]
    }
    def __init__(self, name="CLSSS", *args, **kwargs):
        self.all_possible_configs = [dict(zip(self.params, vals)) for vals in itertools.product(*self.params.values())]
        super().__init__(name, *args, **kwargs)

    def load(self, model_path='sentence-transformers/LaBSE'):
        # model_path = AssetResolver.resolve_path()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.eval()
        self.ready = True

    def run_aggregation(self, request: Dict[str, List[List[MetricAggregate]]]) -> Dict[str, List[MetricAggregate]]:
        """_summary_

        Args:
            metric_aggregates (Dict[str, List[List[MetricAggregate]]]): _description_

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
                aggregate_score = round(np.average(scores, weights=weights), 2)
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

        
    def run_metric(self, request: Dict[str, List[ReferencelessTextGenerationMetricInput]]) -> Dict[str, List[ReferencelessTextGenerationMetricOutput]]:
        """Scoring Function for CLSSS

        Args:
            request (Dict[str, List[ReferencelessTextGenerationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[ReferencelessTextGenerationMetricOutput]]: outuput
        """
        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            data_list, details_list, aggregate_metadata_list = [], [], []
            for current_config in self.all_possible_configs:
                details_dict = {}
                hypotheses, sources = inp.hypotheses, inp.sources
                with torch.no_grad():
                    embed_source = self.model(
                        **self.tokenizer(
                            sources, return_tensors="pt", truncation=True, padding=True
                        )
                    ).pooler_output
                    
                    embed_hyp = self.model(
                        **self.tokenizer(
                            hypotheses, return_tensors="pt", truncation=True, padding=True
                        )
                    ).pooler_output

                    sentences_cos_sim = torch.nn.functional.cosine_similarity(embed_source, embed_hyp).numpy().tolist()
                    seg_scores = [round(c * 100, 2) for c in sentences_cos_sim]
                    sys_score = round(np.average(seg_scores), 2)
                    details_dict['sentence-level'] = [round(s, 4) for s in seg_scores]
                    details_dict['corpus-level'] = round(sys_score, 4)
                    details_dict['options'] = current_config
                    details_list.append(details_dict)
                    data = {
                        "score" : details_dict['corpus-level'],
                        "options" : current_config
                    }
                    data_list.append(data)
                    aggregation_metadata = {
                        'corpus-level-score' : sys_score,
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
            output_dict = ReferencelessTextGenerationMetricOutput(**{
                "data": data_list,
                "details": details_list,
                "metric_aggregate" : metric_aggregate
            })
            predictions.append(output_dict)
        
        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    asset = CLSSS(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([asset])