## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com

import librosa
import numpy as np
import re
import subprocess

from aixplain.model_schemas.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_schemas.interfaces.metric_models import AudioGenerationMetric
from aixplain.model_schemas.interfaces.asset_resolver import AssetResolver
from aixplain.model_schemas.schemas.metric_input import AudioGenerationMetricInput, MetricAggregate
from aixplain.model_schemas.schemas.metric_output import AudioGenerationMetricOutput
from aixplain.model_schemas.utils.data_utils import download_data

from typing import Dict, List
from pathlib import Path
from tempfile import TemporaryDirectory

MODEL_NOT_FOUND_ERROR = """
    Download model file using command:
    # TODO (krishnadurai): Host this on a public URL
    aws s3 cp --recursive s3://benchmarksdata/models/visqol/ ./external/
"""


class Visqol(AudioGenerationMetric):
    def load(
        self,
    ):

        executable_path = Path(AssetResolver.resolve_path()) / "visqol"
        svm_model_path = Path(AssetResolver.resolve_path()) / "libsvm_nu_svr_model.txt"

        # check if model exists
        if not executable_path.exists() or not svm_model_path.exists():
            raise Exception(MODEL_NOT_FOUND_ERROR)
        self.executable_path = executable_path
        self.svm_model_path = svm_model_path
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

    def run_metric(self, request: Dict[str, List[AudioGenerationMetricInput]], headers: Dict[str, str] = None) -> Dict[str, List[AudioGenerationMetricOutput]]:
        """Scoring Function for Visqol metric

        Args:
            request (Dict[str, List[AudioGenerationMetricInput]]): Input to the metric

        Returns:
            Dict[str, List[AudioGenerationMetricOutput]]: Output of the metric
        """

        if not self.ready:
            raise Exception(f"Visqol model not ready yet. Please call load() first.")

        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            hypotheses, references = inp.hypotheses, inp.references
            seg_scores = []
            aggregate_metadata_list = []
            hyp_lens = []
            for hyp, refs in zip(hypotheses, references):
                with TemporaryDirectory() as tmp_dir:
                    # download hypothesis and referecenses with unique names
                    hyp_file = download_data(hyp, root_dir=Path(tmp_dir))
                    ref_file = download_data(refs[0], root_dir=Path(tmp_dir))

                    hyp_waveform, hyp_sample_rate = librosa.load(hyp_file, sr=16000)
                    hyp_file_len_in_sec = len(hyp_waveform) / hyp_sample_rate

                    # calculate visqol score
                    try:
                        score = self.__calculate_visqol(hyp_file, ref_file)
                    except Exception as e:
                        print(e)
                        score = 0.0
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

    def __calculate_visqol(self, degraded_file_path: str, reference_file_path: str):
        command = [
            self.executable_path,
            "--reference_file",
            reference_file_path,
            "--degraded_file",
            degraded_file_path,
            "--similarity_to_quality_model",
            self.svm_model_path,
        ]

        output = subprocess.check_output(command, timeout=60.0).decode("utf-8")

        mos_pattern = re.compile(r"MOS-LQO:\s*(?P<mos_lqo>[0-9.]+)")
        match = mos_pattern.search(output)
        if match:
            return float(match.group("mos_lqo"))
        else:
            raise Exception("Failed to fetch VISQOL - response does not contain MOS-LQO." " Actual response: {}".format(output))


if __name__ == "__main__":
    model = Visqol(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([model])
