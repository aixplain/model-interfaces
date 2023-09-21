__author__='aiXplain'

from aixplain.model_interfaces.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_interfaces.interfaces.metric_models import ClassificationMetric
from aixplain.model_interfaces.interfaces.asset_resolver import AssetResolver
from aixplain.model_interfaces.schemas.metric_input import ClassificationMetricInput
from aixplain.model_interfaces.schemas.metric_output import ClassificationMetricOutput
from sklearn.metrics import hamming_loss
from typing import Dict, List

class HammingLoss(ClassificationMetric):
    def run_metric(self, request: Dict[str, List[ClassificationMetricInput]]) -> Dict[str, List[ClassificationMetricOutput]]:
        """Scoring function

        Args:
            request (Dict[str, List[ClassificationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[ClassificationMetricOutput]]: _description_
        """
        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            return_dict = {}
            metric = inp.metric.lower().strip()
            hypotheses, references = inp.hypotheses, inp.references
            
            # corpus level scores
            score = hamming_loss(references, hypotheses)
            score = round(score, 4)
            return_dict['corpus-level'] = score

            output_dict = ClassificationMetricOutput(**{
                "data": return_dict['corpus-level'],
                "details": return_dict
            })
            predictions.append(output_dict)
        
        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    model = HammingLoss(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([model])