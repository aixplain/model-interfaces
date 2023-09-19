__author__ = "aiXplain"

from aixplain.model_schemas.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_schemas.interfaces.metric_models import NamedEntityRecognitionMetric
from aixplain.model_schemas.interfaces.asset_resolver import AssetResolver
from aixplain.model_schemas.schemas.metric_input import NamedEntityRecognitionMetricInput, NamedEntityRecognitionElement
from aixplain.model_schemas.schemas.metric_output import NamedEntityRecognitionMetricOutput
from typing import Dict, List


class SemEvalNERMetric(NamedEntityRecognitionMetric):
    """Class for evaluating Named Entity Recognition (NER) models as defined in the SemEval 2013 - 9.1 task.

    Source:
        https://github.com/MantisAI/nervaluate
        https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    """

    def compute_frequencies(self, hypotheses: List[NamedEntityRecognitionElement], references: List[NamedEntityRecognitionElement]):
        """Compute frequencies according to SemEval-13 - task 9.1, where named entities are labeled according to
           correct, incorrect, partial, missed and spurious in the classes type, partial, exact and strict

        Args:
            hypotheses (List[NamedEntityRecognitionElement]): list of hypotheses
            references (List[NamedEntityRecognitionElement]): list of references

        Returns:
            _type_: _description_
        """
        results = {
            "type": {"correct": 0, "incorrect": 0, "partial": 0, "missed": 0, "spurious": 0},
            "partial": {"correct": 0, "incorrect": 0, "partial": 0, "missed": 0, "spurious": 0},
            "exact": {"correct": 0, "incorrect": 0, "partial": 0, "missed": 0, "spurious": 0},
            "strict": {"correct": 0, "incorrect": 0, "partial": 0, "missed": 0, "spurious": 0},
        }

        for (hypotheses_doc, references_doc) in zip(hypotheses, references):
            hypothesis_pos = 0
            are_hypotheses_found = len(hypotheses_doc) * [False]
            for reference in references_doc:
                is_reference_found = False
                ref_start, ref_end = reference.offset, reference.offset + reference.length

                for hyp_idx in range(hypothesis_pos, len(hypotheses_doc)):
                    hypothesis = hypotheses_doc[hyp_idx]
                    hyp_start, hyp_end = hypothesis.offset, hypothesis.offset + hypothesis.length

                    # exact boundary overlap
                    if hyp_start == ref_start and hyp_end == ref_end:
                        if hypothesis.text == reference.text:
                            results["exact"]["correct"] += 1
                            results["partial"]["correct"] += 1

                            if hypothesis.category == reference.category:
                                results["type"]["correct"] += 1
                                results["strict"]["correct"] += 1
                            else:
                                results["strict"]["incorrect"] += 1
                                results["type"]["incorrect"] += 1
                        is_reference_found = True
                        are_hypotheses_found[hyp_idx] = True
                        hypothesis_pos += 1
                        break
                    # partial overlap
                    elif max(0, min(ref_end, hyp_end) - max(ref_start, hyp_start)) > 0:
                        if hypothesis.text in reference.text or reference.text in hypothesis.text:
                            results["partial"]["partial"] += 1
                            results["strict"]["incorrect"] += 1
                            results["exact"]["incorrect"] += 1

                            if hypothesis.category == reference.category:
                                results["type"]["correct"] += 1
                            else:
                                results["type"]["incorrect"] += 1
                        is_reference_found = True
                        are_hypotheses_found[hyp_idx] = True
                        hypothesis_pos += 1
                        break

                if is_reference_found is False:
                    results["type"]["missed"] += 1
                    results["partial"]["missed"] += 1
                    results["exact"]["missed"] += 1
                    results["strict"]["missed"] += 1

            for is_hyp_found in are_hypotheses_found:
                if is_hyp_found is False:
                    results["type"]["spurious"] += 1
                    results["partial"]["spurious"] += 1
                    results["exact"]["spurious"] += 1
                    results["strict"]["spurious"] += 1
        return results

    def run_metric(self, request: Dict[str, List[NamedEntityRecognitionMetricInput]]) -> Dict[str, List[NamedEntityRecognitionMetricOutput]]:
        """Scoring function

        Args:
            request (Dict[str, List[NamedEntityRecognitionMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[NamedEntityRecognitionMetricOutput]]: _description_
        """
        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            frequencies = self.compute_frequencies(hypotheses=inp.hypotheses, references=inp.references)

            results = {}
            for schema in frequencies:
                correct = frequencies[schema]["correct"]
                incorrect = frequencies[schema]["incorrect"]
                partial = frequencies[schema]["partial"]
                missed = frequencies[schema]["missed"]
                spurious = frequencies[schema]["spurious"]

                possible = correct + incorrect + partial + missed
                actual = correct + incorrect + partial + spurious

                if schema in ["strict", "exact"]:
                    precision = correct / actual
                    recall = correct / possible
                else:
                    precision = (correct + (0.5 * partial)) / actual
                    recall = (correct + (0.5 * partial)) / possible
                f1 = 2 * ((precision * recall) / (precision + recall))

                results[schema] = {"possible": possible, "actual": actual, "precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2)}

                frequencies[schema].update(
                    {"possible": possible, "actual": actual, "precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2)}
                )

            output_dict = NamedEntityRecognitionMetricOutput(**{"data": results, "details": frequencies})
            predictions.append(output_dict)

        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    model = SemEvalNERMetric(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([model])
