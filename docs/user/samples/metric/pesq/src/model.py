__author__ = "aiXplain"

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

import librosa
import numpy as np
from pypesq import pesq

from aixplain.model_interfaces.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_interfaces.interfaces.metric_models import AudioGenerationMetric
from aixplain.model_interfaces.interfaces.asset_resolver import AssetResolver
from aixplain.model_interfaces.schemas.metric_input import AudioGenerationMetricInput, MetricAggregate
from aixplain.model_interfaces.schemas.metric_output import AudioGenerationMetricOutput
from aixplain.model_interfaces.utils.data_utils import download_data


class PESQ(AudioGenerationMetric):
    def load(self):

        self.ready = True

    def run_aggregation(self, request: Dict[str, List[List[MetricAggregate]]], headers: Dict[str, str] = None) -> Dict[str, List[MetricAggregate]]:
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
            corpus_sum = 0
            hyp_len_sum = 0
            for sample_info in batch_info:
                sub_sample_info = sample_info["aggregation_metadata"][0]
                corpus_sum += sub_sample_info["corpus-sum"]
                hyp_len_sum += sub_sample_info["hyp-length-sum"]

            aggregate_score = round(corpus_sum / hyp_len_sum, 2)
            aggregate_metadata = {"corpus-sum": corpus_sum, "hyp-length-sum": hyp_len_sum}
            data = {"score": aggregate_score}
            data_list.append(data)
            aggregate_metadata_list.append(aggregate_metadata)
            output_dict = MetricAggregate(
                **{
                    "data": data_list,
                    "aggregation_metadata": aggregate_metadata_list,
                    "supplier": batch_info[0]["supplier"],
                    "metric": batch_info[0]["metric"],
                    "version": batch_info[0]["version"],
                }
            )
            predictions.append(output_dict)

        predict_output = {"aggregates": predictions}
        return predict_output

    def run_metric(self, request: Dict[str, List[AudioGenerationMetricInput]], headers=None) -> Dict[str, List[AudioGenerationMetricOutput]]:
        """Scoring function for PESQ metric.

        Args:
            request (Dict[str, List[AudioGenerationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[AudioGenerationMetricOutput]]: _description_
        """
        if not self.ready:
            raise Exception(f"PESQ model not ready yet. Please call load() first.")

        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            hypotheses, references = inp.hypotheses, inp.references
            aggregate_metadata_list = []
            seg_scores = []
            hyp_lens = []
            for hyp, refs in zip(hypotheses, references):
                with TemporaryDirectory() as tmp_dir:
                    # download hypothesis and referecenses with unique names
                    hyp_file = download_data(hyp, root_dir=Path(tmp_dir))
                    ref_file = download_data(refs[0], root_dir=Path(tmp_dir))

                    hyp_waveform, hyp_sample_rate = librosa.load(hyp_file, sr=16000)
                    ref_waveform, ref_sample_rate = librosa.load(ref_file, sr=16000)

                    hyp_file_len_in_sec = len(hyp_waveform) / hyp_sample_rate

                    assert hyp_sample_rate == ref_sample_rate, "Sample rates of hypothesis and reference are not equal."

                    # compute scores
                    try:
                        score = pesq(ref_waveform, hyp_waveform, fs=hyp_sample_rate, normalize=False)
                    except Exception as e:
                        raise Exception(f"Error while computing PESQ score: {e}")

                seg_scores.append(score)
                hyp_lens.append(hyp_file_len_in_sec)
            sys_score = round(np.average(seg_scores, weights=hyp_lens), 2)
            weighted_sum = round(np.sum(np.multiply(seg_scores, hyp_lens)), 2)

            aggregation_metadata = {
                "corpus-sum": weighted_sum,
                "hyp-length-sum": round(np.sum(hyp_lens), 2),
            }
            aggregate_metadata_list.append(aggregation_metadata)
            return_dict = {
                "corpus-level": round(sys_score, 4),
                "sentence-level": [round(s, 4) for s in seg_scores],
            }
            metric_aggregate = MetricAggregate(
                **{"aggregation_metadata": aggregate_metadata_list, "supplier": inp.supplier, "metric": inp.metric, "version": inp.version}
            )
            output_dict = AudioGenerationMetricOutput(**{"data": return_dict["corpus-level"], "details": return_dict, "metric_aggregate": metric_aggregate})
            predictions.append(output_dict)

        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    metric = PESQ(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([metric])
