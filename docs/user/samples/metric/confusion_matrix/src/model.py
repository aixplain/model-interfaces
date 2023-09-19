__author__='aiXplain'

from aixplain.model_schemas.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_schemas.interfaces.metric_models import ClassificationMetric
from aixplain.model_schemas.interfaces.asset_resolver import AssetResolver
from aixplain.model_schemas.schemas.metric_input import ClassificationMetricInput
from aixplain.model_schemas.schemas.metric_output import ClassificationMetricOutput
from sklearn.metrics import confusion_matrix
from typing import Dict, List

class ConfusionMatrix(ClassificationMetric):
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
            
            labels = sorted(list(set(references)))
            score = confusion_matrix(references, hypotheses, labels=labels).tolist()
            return_dict['labels'] = labels
            return_dict['corpus-level'] = score

            output_dict = ClassificationMetricOutput(**{
                "data": return_dict['corpus-level'],
                "details": return_dict
            })
            predictions.append(output_dict)
        
        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    model = ConfusionMatrix(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([model])