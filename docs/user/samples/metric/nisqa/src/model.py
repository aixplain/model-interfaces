## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com

import numpy as np
import librosa as librosa

from aixplain.model_schemas.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_schemas.interfaces.metric_models import ReferencelessAudioGenerationMetric
from aixplain.model_schemas.interfaces.asset_resolver import AssetResolver
from aixplain.model_schemas.schemas.metric_input import ReferencelessAudioGenerationMetricInput, MetricAggregate
from aixplain.model_schemas.schemas.metric_output import (
    ReferencelessAudioGenerationMetricOutput,
)
from aixplain.model_schemas.utils.data_utils import download_data

from typing import Dict, List
from pathlib import Path
from tempfile import TemporaryDirectory
from .external.nisqa_model import NisqaModel

MODEL_NOT_FOUND_ERROR = """
    Download model file using command:
    # TODO (krishnadurai): Host this on a public URL
    aws s3 cp --recursive s3://benchmarksdata/models/nisqa/ ./external/
"""


class Nisqa(ReferencelessAudioGenerationMetric):
    def load(self):
        model_path = Path(AssetResolver.resolve_path()) / "nisqa_tts.tar"
        # check if model exists
        if not model_path.exists():
            raise Exception(MODEL_NOT_FOUND_ERROR)
        self.config = {
            "mode": "predict_file",
            "tr_device": "cpu",
            "output_dir": "results",
            "pretrained_model": str(model_path.resolve()),
            "mode": "predict_file",
        }
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

    def run_metric(
        self, request: Dict[str, List[ReferencelessAudioGenerationMetricInput]], headers: Dict[str, str] = None
    ) -> Dict[str, List[ReferencelessAudioGenerationMetricOutput]]:
        """Scoring Function for Visqol metric

        Args:
            request (Dict[str, List[ReferencelessAudioGenerationMetricInput]]): Input to the metric

        Returns:
            Dict[str, List[ReferencelessAudioGenerationMetricOutput]]: Output of the metric
        """

        if not self.ready:
            raise Exception(f"Visqol model not ready yet. Please call load() first.")

        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            (hypotheses,) = inp.hypotheses
            seg_scores = []
            aggregate_metadata_list = []
            hyp_lens = []
            for hyp in [hypotheses]:
                with TemporaryDirectory() as tmp_dir:
                    # download hypothesis and source with unique names
                    hyp_file = download_data(hyp, root_dir=Path(tmp_dir))

                    hyp_waveform, hyp_sample_rate = librosa.load(hyp_file, sr=16000)
                    hyp_file_len_in_sec = len(hyp_waveform) / hyp_sample_rate

                    # calculate nisqa score
                    score = self.__calculate_nisqa(hyp_file)

                seg_scores.append(score)
                hyp_lens.append(hyp_file_len_in_sec)

            sys_score = round(np.average(seg_scores), 2)
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

            output_dict = ReferencelessAudioGenerationMetricOutput(
                **{"data": return_dict["corpus-level"], "details": return_dict, "metric_aggregate": metric_aggregate}
            )
            predictions.append(output_dict)

        predict_output = {"scores": predictions}
        return predict_output

    def __calculate_nisqa(self, degraded_file_path: str):
        self.config["deg"] = degraded_file_path
        try:
            nisqa = NisqaModel(self.config)
            score = nisqa.predict()
            return score
        except Exception as e:
            raise Exception(f"Error while calculating NISQA score: {e}")


if __name__ == "__main__":
    model = Nisqa(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([model])
