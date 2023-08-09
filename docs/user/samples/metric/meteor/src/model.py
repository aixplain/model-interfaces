__author__ = "aiXplain"

import aixplain.aixplain_models.utils.metric_utils as utils
from aixplain.aixplain_models.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.aixplain_models.interfaces.metric_models import TextGenerationMetric
from aixplain.aixplain_models.interfaces.asset_resolver import AssetResolver
from aixplain.aixplain_models.schemas.metric_input import TextGenerationMetricInput, MetricAggregate
from aixplain.aixplain_models.schemas.metric_output import TextGenerationMetricOutput
import evaluate
import numpy as np
import aixplain.aixplain_models.utils.metric_utils as metric_utils
import itertools
from typing import Dict, List


class Meteor(TextGenerationMetric):
    params = {"casing": ["Cased", "Uncased"], "punctuation": ["Punctuated", "Not Punctuated"]}

    def __init__(self, name="meteor", *args, **kwargs):
        self.all_possible_configs = [dict(zip(self.params, vals)) for vals in itertools.product(*self.params.values())]
        super().__init__(name, *args, **kwargs)

    def load(self, model_name: str = "meteor", gpus: int = 1):
        model_path = AssetResolver.resolve_path()
        self.model = evaluate.load(model_name)
        self.gpus = gpus
        self.ready = True

    def run_aggregation(self, request: Dict[str, List[List[MetricAggregate]]]) -> Dict[str, List[MetricAggregate]]:
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
                scores, weights = [], []
                for sample_info in batch_info:
                    sub_sample_info = sample_info.aggregation_metadata[sub_sample_index]
                    scores.append(sub_sample_info["corpus-level-score"])
                    weights.append(sub_sample_info["hypotheses_count"])
                aggregate_score = round(np.average(scores, weights=weights), 2)
                aggregate_metadata = {"corpus-level-score": aggregate_score, "hypotheses_count": np.sum(weights), "options": sub_sample_info["options"]}
                data = {"score": aggregate_score, "options": sub_sample_info["options"]}
                data_list.append(data)
                aggregate_metadata_list.append(aggregate_metadata)
            output_dict = MetricAggregate(
                **{
                    "data": data_list,
                    "aggregation_metadata": aggregate_metadata_list,
                    "supplier": batch_info[0].supplier,
                    "metric": batch_info[0].metric,
                    "version": batch_info[0].version,
                }
            )
            predictions.append(output_dict)

        predict_output = {"aggregates": predictions}
        return predict_output

    def run_metric(self, request: Dict[str, List[TextGenerationMetricInput]]) -> Dict[str, List[TextGenerationMetricOutput]]:
        """Scoring function

        Args:
            request (Dict[str, List[TextGenerationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[TextGenerationMetricOutput]]: _description_
        """
        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            data_list, details_list, aggregate_metadata_list = [], [], []
            for current_config in self.all_possible_configs:
                details_dict = {}
                hypotheses, references = inp.hypotheses.copy(), inp.references.copy()
                transposed_references = utils.transpose(references)
                if current_config["casing"] == "Uncased":
                    hypotheses = metric_utils.lowercase(hypotheses)
                    for j, reference in enumerate(references):
                        references[j] = metric_utils.lowercase(reference)
                if current_config["punctuation"] == "Not Punctuated":
                    hypotheses = metric_utils.remove_punctuation(hypotheses)
                    for j, reference in enumerate(references):
                        references[j] = metric_utils.remove_punctuation(reference)

                # sentence level scores
                scores = []
                for i in range(len(hypotheses)):
                    try:
                        hyp, refs = hypotheses[i], references[i]
                        refs = refs if type(refs) is list else [refs]
                        results = self.model.compute(predictions=[hyp], references=[refs])
                        score = round(results["meteor"] * 100, 2)
                        scores.append(score)
                    except Exception as e:
                        scores.append(0.0)
                details_dict["sentence-level"] = scores

                # corpus level scores
                results = self.model.compute(predictions=hypotheses, references=references)
                score = round(results["meteor"] * 100, 2)
                details_dict["corpus-level"] = score
                details_dict["options"] = current_config
                details_list.append(details_dict)
                data = {"score": details_dict["corpus-level"], "options": current_config}
                data_list.append(data)
                aggregation_metadata = {"corpus-level-score": details_dict["corpus-level"], "hypotheses_count": len(hypotheses), "options": current_config}
                aggregate_metadata_list.append(aggregation_metadata)

            metric_aggregate = MetricAggregate(
                **{"aggregation_metadata": aggregate_metadata_list, "supplier": inp.supplier, "metric": inp.metric, "version": inp.version}
            )

            output_dict = TextGenerationMetricOutput(**{"data": data_list, "details": details_list, "metric_aggregate": metric_aggregate})
            predictions.append(output_dict)

        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    model = Meteor(AssetResolver.model_uri())
    AixplainModelServer(workers=1).start([model])
