## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com

import numpy as np

from aixplain.model_interfaces.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_interfaces.interfaces.metric_models import AudioGenerationMetric
from aixplain.model_interfaces.interfaces.asset_resolver import AssetResolver
from aixplain.model_interfaces.schemas.metric_input import AudioGenerationMetricInput, MetricAggregate
from aixplain.model_interfaces.schemas.metric_output import AudioGenerationMetricOutput
from aixplain.model_interfaces.utils.data_utils import download_data

from typing import Dict, List
from pathlib import Path
from tempfile import TemporaryDirectory

# Load libraries
from .external.warpq import WARPQ
import librosa
import soundfile as sf

MODEL_NOT_FOUND_ERROR = """
    Download model file using command:
    # TODO (krishnadurai): Host this on a public URL
    aws s3 cp --recursive s3://benchmarksdata/models/warpq/ ./external/
"""

SAMPLING_RATE = 16000
N_MFCC = 13
FMAX = 5000
PATCH_SIZE = 0.4
SIGMA = [[1, 0], [0, 3], [1, 3]]
APPLY_VAD = True
GETPLOTS = False


class Warpq(AudioGenerationMetric):
    def load(
        self,
    ):

        mode = "predict_file"
        sr = 16000
        n_mfcc = 13
        fmax = 5000
        patch_size = 0.4
        sigma = [[1, 0], [0, 3], [1, 3]]
        apply_vad = True
        getPlots = False

        self.model_path = Path(AssetResolver.resolve_path()) / "RandomForest" / "Genspeech_TCDVoIP_PSup23.zip"

        # check if model exists
        if not self.model_path.exists():
            raise Exception(MODEL_NOT_FOUND_ERROR)

        # Object of WARP-Q class
        self.model = WARPQ(self.model_path, sr, n_mfcc, fmax, patch_size, sigma, apply_vad)
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

                    # correct the sampling rate of the files if needed
                    if self.__get_sampling_rate(hyp_file) != SAMPLING_RATE:
                        hyp_file = self.__resample(hyp_file, SAMPLING_RATE)
                    if self.__get_sampling_rate(ref_file) != SAMPLING_RATE:
                        ref_file = self.__resample(ref_file, SAMPLING_RATE)

                    # calculate warpq score
                    score = self.__calculate_warpq(hyp_file, ref_file)
                    hyp_lens.append(hyp_file_len_in_sec)

                seg_scores.append(score)

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

            output_dict = AudioGenerationMetricOutput(**{"data": return_dict["corpus-level"], "details": return_dict, "metric_aggregate": metric_aggregate})
            predictions.append(output_dict)

        predict_output = {"scores": predictions}
        return predict_output

    def __calculate_warpq(self, degraded_file_path: str, reference_file_path: str):
        try:
            warpq_rawScore, warpq_mappedScore = self.model.evaluate(reference_file_path, degraded_file_path)
            return warpq_mappedScore
        except Exception as e:
            raise Exception("Failed to calculate WARP-Q score. Please check your input files. ERROR: {}".format(e))

    def __get_sampling_rate(self, file_path: str):
        sr = librosa.get_samplerate(file_path)
        return sr

    def __resample(self, file_path: str, sr: int):
        y, sr = librosa.load(file_path, sr=sr)
        sf.write(file_path, y, sr)
        return file_path


if __name__ == "__main__":
    model = Warpq(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([model])
