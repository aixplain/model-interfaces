__author__='aiXplain'

from aixplain.model_schemas.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_schemas.interfaces.metric_models import ClassificationMetric
from aixplain.model_schemas.interfaces.asset_resolver import AssetResolver
from aixplain.model_schemas.schemas.metric_input import ClassificationMetricInput
from aixplain.model_schemas.schemas.metric_output import ClassificationMetricOutput
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, hamming_loss, precision_score, recall_score
from typing import Dict, List

class SklearnClassificationMetrics(ClassificationMetric):
    def run_metric(self, request: Dict[str, List[ClassificationMetricInput]]) -> Dict[str, List[ClassificationMetricOutput]]:
        """Scoring function

        Args:
            request (Dict[str, List[SequenceClassificationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[SequenceClassificationMetricOutput]]: _description_
        """
        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            return_dict = {}
            metric = inp.metric.lower().strip()
            hypotheses, references = inp.hypotheses, inp.references
            
            # corpus level scores
            if metric == 'accuracy':
                score = accuracy_score(references, hypotheses)
                score = round(score, 4)
            elif metric == 'f1':
                # if binary, binary f1-score. Otherwise, weighted
                if len(set(hypotheses)) == 2:
                    score = f1_score(references, hypotheses, average="binary")
                else:
                    score = f1_score(references, hypotheses, average="weighted")
                score = round(score, 4)
            elif metric == 'f1-micro':
                score = f1_score(references, hypotheses, average="micro")
                score = round(score, 4)
            elif metric == 'f1-macro':
                score = f1_score(references, hypotheses, average="macro")
                score = round(score, 4)
            elif metric == 'f1-weighted':
                score = f1_score(references, hypotheses, average="weighted")
                score = round(score, 4)
            elif metric == 'f1-samples':
                score = f1_score(references, hypotheses, average="samples")
                score = round(score, 4)
            elif metric == 'precision':
                # if binary, binary precision. Otherwise, weighted
                if len(set(hypotheses)) == 2:
                    score = precision_score(references, hypotheses, average="binary")
                else:
                    score = precision_score(references, hypotheses, average="weighted")
                score = round(score, 4)
            elif metric == 'precision-micro':
                score = precision_score(references, hypotheses, average="micro")
                score = round(score, 4)
            elif metric == 'precision-macro':
                score = precision_score(references, hypotheses, average="macro")
                score = round(score, 4)
            elif metric == 'precision-weighted':
                score = precision_score(references, hypotheses, average="weighted")
                score = round(score, 4)
            elif metric == 'precision-samples':
                score = precision_score(references, hypotheses, average="samples")
                score = round(score, 4)
            elif metric == 'recall':
                # if binary, binary recall. Otherwise, weighted
                if len(set(hypotheses)) == 2:
                    score = recall_score(references, hypotheses, average="binary")
                else:
                    score = recall_score(references, hypotheses, average="weighted")
                score = round(score, 4)
            elif metric == 'recall-micro':
                score = recall_score(references, hypotheses, average="micro")
                score = round(score, 4)
            elif metric == 'recall-macro':
                score = recall_score(references, hypotheses, average="macro")
                score = round(score, 4)
            elif metric == 'recall-weighted':
                score = recall_score(references, hypotheses, average="weighted")
                score = round(score, 4)
            elif metric == 'recall-samples':
                score = recall_score(references, hypotheses, average="samples")
                score = round(score, 4)
            elif metric == 'confusion_matrix':
                labels = sorted(list(set(references)))
                score = confusion_matrix(references, hypotheses, labels=labels).tolist()
                return_dict['labels'] = labels
            elif metric == 'hamming_loss':
                score = hamming_loss(references, references)
                score = round(score, 4)
            else:
                score = classification_report(references, hypotheses, output_dict=True)
                for label in score:
                    if isinstance(score[label], dict):
                        for metric in score[label]:
                            score[label][metric] = round(score[label][metric], 4)
                    else:
                        score[label] = round(score[label], 4)
            return_dict['corpus-level'] = score

            output_dict = ClassificationMetricOutput(**{
                "data": return_dict['corpus-level'],
                "details": return_dict
            })
            predictions.append(output_dict)
        
        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    model = SklearnClassificationMetrics(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([model])