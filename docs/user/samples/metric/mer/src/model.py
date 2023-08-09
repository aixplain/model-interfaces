__author__='aiXplain'

import aixplain.aixplain_models.utils.metric_utils as utils
from aixplain.aixplain_models.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.aixplain_models.interfaces.metric_models import TextGenerationMetric
from aixplain.aixplain_models.interfaces.model_resolver import ModelResolver
from aixplain.aixplain_models.schemas.metric_input import TextGenerationMetricInput
from aixplain.aixplain_models.schemas.metric_output import TextGenerationMetricOutput
import torchmetrics 
from torchmetrics.metric import Metric as BaseTorchMetric
import numpy as np
import math
from typing import Dict, List

class MER(TextGenerationMetric):
    higher_is_better_metric_info = {
        "wip" : True,
        "wil" : False,
        "mer" : False
    }

    default_metric_info = {
        "wip" : 0.0,
        "wil" : 1.0,
        "mer" : 1.0
    }

    def get_class_for_metric(self, metric_name:str) -> BaseTorchMetric:
        if metric_name == 'wip':
            Metric = torchmetrics.WordInfoPreserved
        elif metric_name == 'wil':
            Metric = torchmetrics.WordInfoLost
        elif metric_name == 'mer':
            Metric = torchmetrics.MatchErrorRate
        return Metric

    def get_object_for_multi_reference(self, metric_name:str , hypothesis: str, references: List[str]) -> BaseTorchMetric:
        """Simplify to best torchmetric object to handle multiple references

        Args:
            metric_name (str): Metric to be calculated
            hypothesis (str): Hypthesis sentence
            references (List[str]): List of references

        Returns:
            BaseTorchMetric: best torch metric object
        """
        Metric = self.get_class_for_metric(metric_name)
        best_metric_object = None
        for reference in references:
            metric_object = Metric()
            metric_object.update(hypothesis, reference)
            if best_metric_object is None:
                best_metric_object = metric_object
            else:
                if self.higher_is_better_metric_info[metric_name]:
                    if best_metric_object.errors < metric_object.errors:
                        best_metric_object = metric_object
                else:
                    if best_metric_object.errors > metric_object.errors:
                        best_metric_object = metric_object
        return best_metric_object


    def get_corpus_score_from_list(self, metric_object_list: List[BaseTorchMetric]) -> float:
        """Get corpus level scores from sentence level metric objects

        Args:
            metric_object_list (List[BaseTorchMetric]): List of setence level metric objects

        Returns:
            float: corpus level score
        """
        metric_object_list = [metric_object for metric_object in metric_object_list if metric_object is not None]
        final_metric_object = metric_object_list[0]
        for metric_object in metric_object_list[1::]:
            obj_state = metric_object.__getstate__()
            final_metric_object._reduce_states(obj_state)
        return float(final_metric_object.compute())
    

    def run_aggregation(self, api_outputs: Dict[str, List[List[TextGenerationMetricOutput]]]) -> Dict[str, List[TextGenerationMetricOutput]]:
        """Aggregation function to aggregate previous computed scores

        Args:
            api_outputs (Dict[str, List[List[TextGenerationMetricOutput]]]): outputs of the APIs

        Returns:
            Dict[str, List[TextGenerationMetricOutput]]: _description_
        """
        metric = "mer"
        predictions = []
        outputs = api_outputs["instances"]
        for output in outputs:
            all_state_dicts = []
            for score_info in output:
                all_state_dicts += score_info.details['object-state-dict-list']
            metric_object_list = []
            for state_dict in all_state_dicts:
                if state_dict is not None:
                    metric_object = self.get_class_for_metric(metric)()
                    metric_object.__setstate__(state_dict)
                    metric_object_list.append(metric_object)
            corpus_score = self.get_corpus_score_from_list(metric_object_list)
            corpus_score = self.default_metric_info[metric] if math.isnan(corpus_score) else corpus_score
            output_dict = TextGenerationMetricOutput(**{
                "data": round(corpus_score*100,2),
                "details": {}
            })
            predictions.append(output_dict)
        
        predict_output = {"scores": predictions}
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
            return_dict = {}
            hypotheses, references = inp.hypotheses, inp.references
            transposed_references = utils.transpose(references)
            metric = "mer"
            metric_object_list = []
            for i in range(len(hypotheses)):
                try:
                    hyp, refs = hypotheses[i], references[i]
                    refs = refs if type(refs) is list else [refs]
                    metric_object_list.append(self.get_object_for_multi_reference(metric, hyp, refs))
                except:
                    metric_object_list.append(None)
            
            # sentence level scores
            return_dict['sentence-level'] = [round(float(metric_object.compute())*100,2) if metric_object is not None else None for metric_object in metric_object_list ]
            return_dict['sentence-level'] = [self.default_metric_info[metric] if math.isnan(sentence_score) else sentence_score for sentence_score in return_dict['sentence-level']]
            # corpus level scores
            corpus_score = self.get_corpus_score_from_list(metric_object_list)
            corpus_score = self.default_metric_info[metric] if math.isnan(corpus_score) else corpus_score
            return_dict['corpus-level'] = round(corpus_score*100,2)
            
            return_dict['object-state-dict-list'] = [metric_object.__getstate__() if metric_object is not None else None for metric_object in metric_object_list ]

            output_dict = TextGenerationMetricOutput(**{
                "data": return_dict['corpus-level'],
                "details": return_dict
            })
            predictions.append(output_dict)
        
        predict_output = {"scores": predictions}
        return predict_output



if __name__ == "__main__":
    model = MER(ModelResolver.model_uri())
    AixplainModelServer(workers=1).start([model])