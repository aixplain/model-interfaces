## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com

import numpy as np

from aixplain.model_interfaces.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_interfaces.interfaces.metric_models import ReferencelessAudioGenerationMetric
from aixplain.model_interfaces.interfaces.asset_resolver import AssetResolver
from aixplain.model_interfaces.schemas.metric_input import ReferencelessAudioGenerationMetricInput, MetricAggregate
from aixplain.model_interfaces.schemas.metric_output import (
    ReferencelessAudioGenerationMetricOutput,
)
from aixplain.model_interfaces.utils.data_utils import download_data

from typing import Dict, List
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


MODEL_NOT_FOUND_ERROR = """
    Download model file using command:
    # TODO (krishnadurai): Host this on a public URL
    aws s3 cp --recursive s3://benchmarksdata/models/dnsmos/ ./external/
"""


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01


class DNSMOS(ReferencelessAudioGenerationMetric):
    def load(self):
        self.is_personalized_MOS = False  # TODO (krishnadurai): make this configurable
        model_path = Path(AssetResolver.resolve_path()) / "sig_bak_ovr.onnx"
        if self.is_personalized_MOS:
            model_path = Path(AssetResolver.resolve_path()) / "psig_bak_ovr.onnx"

        # check if model exists
        if not model_path.exists():
            raise Exception(MODEL_NOT_FOUND_ERROR)

        self.onnx_sess = ort.InferenceSession(str(model_path.resolve()))
        self.ready = True

    def __get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

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
        """Scoring Function for DNSMOS

        Args:
            request (Dict[str, List[ReferencelessAudioGenerationMetricInput]]): Input to the metric

        Returns:
            Dict[str, List[ReferencelessAudioGenerationMetricOutput]]: Output of the metric
        """

        if not self.ready:
            raise Exception(f"DNSMOS model not ready yet. Please call load() first.")

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

                    score = self.__calculate_dnsmos(fpath=hyp_file, sampling_rate=SAMPLING_RATE, is_personalized_MOS=self.is_personalized_MOS)
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

    def __calculate_dnsmos(self, fpath, sampling_rate, is_personalized_MOS):

        try:
            aud, input_fs = sf.read(fpath)
            fs = sampling_rate
            if input_fs != fs:
                audio = librosa.resample(aud, input_fs, fs)
            else:
                audio = aud
            len_samples = int(INPUT_LENGTH * fs)
            while len(audio) < len_samples:
                audio = np.append(audio, audio)

            num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
            hop_len_samples = fs
            predicted_mos_ovr_seg = []

            for idx in range(num_hops):
                audio_seg = audio[int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)]
                if len(audio_seg) < len_samples:
                    continue

                input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
                oi = {"input_1": input_features}
                mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
                mos_sig, mos_bak, mos_ovr = self.__get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)
                predicted_mos_ovr_seg.append(mos_ovr)
            return np.mean(predicted_mos_ovr_seg)
        except Exception as e:
            raise Exception(f"Error while calculating DNSMOS score: {e}")


if __name__ == "__main__":
    metric = DNSMOS(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([metric])
