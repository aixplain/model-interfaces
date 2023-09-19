__author__='aiXplain'

from typing import Dict, List
import itertools

import sacrebleu

from aixplain.model_schemas.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_schemas.interfaces.metric_models import TextGenerationMetric
from aixplain.model_schemas.schemas.metric_input import TextGenerationMetricInput, MetricAggregate
from aixplain.model_schemas.interfaces.asset_resolver import AssetResolver
from aixplain.model_schemas.schemas.metric_output import TextGenerationMetricOutput
import aixplain.model_schemas.utils.metric_utils as metric_utils

class BLEU(TextGenerationMetric):
    params = {
        "casing" : ["Cased", "Uncased"],
        "punctuation" : ["Punctuated", "Not Punctuated"]
    }

    def __init__(self, name="BLEU", *args, **kwargs):
        self.all_possible_configs = [dict(zip(self.params, vals)) for vals in itertools.product(*self.params.values())]
        super().__init__(name, *args, **kwargs)
        
    def run_aggregation(
        self,
        request: Dict[str, List[List[MetricAggregate]]],
        headers: Dict[str, str] = None
    ) -> Dict[str, List[MetricAggregate]]:
        """Aggregation function to aggregate previous computed scores

        Args:
            api_outputs (Dict[str, List[List[MetricAggregate]]]): outputs of the APIs

        Returns:
            Dict[str, List[MetricAggregate]]: _description_
        """
        predictions = []
        batches = request["instances"]
        for batch_info in batches:
            aggregate_metadata_list, data_list = [], []
            num_sub_samples = len(batch_info[0].aggregation_metadata)
            for sub_sample_index in range(num_sub_samples):
                all_stats, total_hyp_count = [], 0
                for sample_info in batch_info:
                    sub_sample_info = sample_info.aggregation_metadata[sub_sample_index]
                    all_stats.append(sub_sample_info["corpus_stats"])
                    total_hyp_count += sub_sample_info['hypotheses_count']
                bleu = sacrebleu.metrics.bleu.BLEU()
                corpus_stats = sacrebleu.utils.sum_of_lists(all_stats)
                aggregate_score = round(bleu._compute_score_from_stats(corpus_stats).score, 2)
                aggregate_metadata = {
                    "corpus_stats" : corpus_stats,
                    "hypotheses_count" : total_hyp_count,
                    "options" : sub_sample_info['options']
                }
                data = {
                    "score" : aggregate_score,
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


    def run_metric(
        self,
        request: Dict[str, List[TextGenerationMetricInput]],
        headers: Dict[str, str] = None
    ) -> Dict[str, List[TextGenerationMetricOutput]]:
        """BLEU scoring function based on sacrebleu

        Args:
            request (Dict[str, List[TextGenerationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[TextGenerationMetricOutput]]: _description_
        """
        inputs = request["instances"]

        predictions = []

        for inp in inputs:
            data_list, details_list, aggregate_metadata_list = [], [], []
            metric = inp.metric.lower().strip()
            for current_config in self.all_possible_configs:
                details_dict = {}
                hypotheses, references = inp.hypotheses.copy(), inp.references.copy()
                if current_config["casing"] == "Uncased":
                    hypotheses = metric_utils.lowercase(hypotheses)
                    for j, reference in enumerate(references):
                        references[j] = metric_utils.lowercase(reference)
                if current_config["punctuation"] == "Not Punctuated":
                    hypotheses = metric_utils.remove_punctuation(hypotheses)
                    for j, reference in enumerate(references):
                        references[j] = metric_utils.remove_punctuation(reference)

                transposed_references = metric_utils.transpose(references)

                # sentence level scores
                scores = []
                all_stats = []
                for i in range(len(hypotheses)):
                    try:
                        bleu = sacrebleu.metrics.bleu.BLEU()
                        stats = bleu._extract_corpus_statistics([hypotheses[i]], [[refs] if type(refs)!=list else refs for refs in references[i]])
                        all_stats += stats
                        score = round(bleu._aggregate_and_compute(stats).score, 2)
                        scores.append(score)
                    except:
                        scores.append(0.0)
                details_dict['sentence-level'] = scores
            
                # corpus level scores
                bleu = sacrebleu.metrics.bleu.BLEU()
                corpus_stats = sacrebleu.utils.sum_of_lists(all_stats)
                score = bleu._compute_score_from_stats(corpus_stats).score
                score = round(score, 2)
                details_dict['corpus-level'] = score
                details_dict['options'] = current_config
                details_list.append(details_dict)
                data = {
                    "score" : score,
                    "options" : current_config
                }
                data_list.append(data)
                aggregation_metadata = {
                    'corpus_stats' : corpus_stats,
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
    metric = BLEU(AssetResolver.asset_uri())
    metric.load()
    AixplainModelServer().start([metric])