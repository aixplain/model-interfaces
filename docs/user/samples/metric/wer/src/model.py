__author__='aiXplain'

from aixplain_models.interfaces.aixplain_model_server import AixplainModelServer
from aixplain_models.interfaces.metric_models import TextGenerationMetric
from aixplain_models.interfaces.asset_resolver import AssetResolver
from aixplain_models.schemas.metric_input import TextGenerationMetricInput
from aixplain_models.schemas.metric_output import TextGenerationMetricOutput
from jiwer import compute_measures
from typing import Dict, List

class WER(TextGenerationMetric):
    def __init__(self, name="WER", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def run_aggregation(self, api_outputs: Dict[str, List[List[TextGenerationMetricOutput]]]) -> Dict[str, List[TextGenerationMetricOutput]]:
        """Aggregation function to aggregate previous computed scores

        Args:
            api_outputs (Dict[str, List[List[TextGenerationMetricOutput]]]): outputs of the APIs

        Returns:
            Dict[str, List[TextGenerationMetricOutput]]: _description_
        """
        predictions = []
        outputs = api_outputs["instances"]
        for output in outputs:
            total_error, ref_lens = 0, 0
            for score_info in output:
                error = score_info.details['wed']['corpus-level']
                ref_len = score_info.details['wer']["corpus-level"]["ref_len"]

                total_error += error
                ref_lens += ref_len
        
            total_score_dict = {
                "wer" : round(total_error/ref_lens, 2),
                "wed" : total_error
            }
            output_dict = TextGenerationMetricOutput(**{
                "data": total_score_dict['wer'],
                "details": total_score_dict
            })
            predictions.append(output_dict)
        
        predict_output = {"scores": predictions}
        return predict_output

    def run_metric(self, request: Dict[str, List[TextGenerationMetricInput]]) -> Dict[str, List[TextGenerationMetricOutput]]:
        """Word Error Rate Scoring function

        Args:
            request (Dict[str, List[TextGenerationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[TextGenerationMetricOutput]]: _description_
        """
        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            return_dict = {}

            hypotheses, references = inp.hypotheses, inp.references

            return_dict = {
                "wer": {
                    "sentence-level": []
                },
                "wed": {
                    "sentence-level": []
                }
            }
            overall_min_errors, total_words = 0, 0
            for hypothesis, reference in zip(hypotheses, references):
                min_errors = float('inf')
                ref_words = 0
                for ref in reference:
                    try:
                        measures = compute_measures(ref, hypothesis)
                        errors = measures["substitutions"] + measures["deletions"] + measures["insertions"]
                        n_words = measures["substitutions"] + measures["deletions"] + measures["hits"]
                    except:
                        n_words = max([len(hypothesis.split()), len(ref.split())])
                        errors = n_words
                    if errors < min_errors:
                        min_errors = errors
                        ref_words = n_words
                
                overall_min_errors += min_errors
                total_words += ref_words
                score = 100 * (min_errors / ref_words)
                return_dict['wer']['sentence-level'].append(float(round(score, 2)))
                return_dict['wed']['sentence-level'].append(min_errors)

            score = 100 * (overall_min_errors/total_words)
            return_dict['wer']['corpus-level'] = {
                "score" : float(round(score, 2)),
                "ref_len" : total_words
                }
            return_dict['wed']['corpus-level'] = overall_min_errors

            output_dict = TextGenerationMetricOutput(**{
                "data": return_dict["wer"]['corpus-level'],
                "details": return_dict
            })
            predictions.append(output_dict)
        
        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    model = WER(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([model])