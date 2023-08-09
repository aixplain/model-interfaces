## created by ahmetgunduz at 20221209 05:45.
##
## email: ahmetgunduz@aixplain.com

import numpy as np

from aixplain.aixplain_models.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.aixplain_models.interfaces.metric_models import ReferencelessAudioGenerationMetric
from aixplain.aixplain_models.interfaces.asset_resolver import AssetResolver
from aixplain.aixplain_models.schemas.metric_input import ReferencelessAudioGenerationMetricInput, MetricAggregate
from aixplain.aixplain_models.schemas.metric_output import (
    ReferencelessAudioGenerationMetricOutput,
)
from aixplain.aixplain_models.utils.data_utils import download_data

from typing import Dict, List
from pathlib import Path
from tempfile import TemporaryDirectory
from .external.noresqa import NORESQA

import torch
import librosa as librosa
import numpy as np
from .external.utils import feats_loading

MODEL_NOT_FOUND_ERROR = """
    Download model file using command:
    # TODO (krishnadurai): Host this on a public URL
    aws s3 cp --recursive s3://benchmarksdata/models/noresqa_mos/ ./external/
"""


class NoresqaMos(ReferencelessAudioGenerationMetric):
    def load(self):
        self.metric_type = 1
        self.gpu_id = 0
        self.mode = "file"
        # current dir + external
        self.nmr = Path(__file__).parent / "external" / "clean.wav"

        CONFIG_PATH = Path(AssetResolver.resolve_path()) / "wav2vec_small.pt"

        if self.metric_type == 0:
            model_path = Path(AssetResolver.resolve_path()) / "model_noresqa.pth"
            state = torch.load(model_path, map_location="cpu")["state_base"]
        elif self.metric_type == 1:
            model_path = Path(AssetResolver.resolve_path()) / "model_noresqa_mos.pth"
            state = torch.load(model_path, map_location="cpu")["state_dict"]

        if not model_path.exists() or not CONFIG_PATH.exists():
            raise Exception(MODEL_NOT_FOUND_ERROR)

        # Noresqa model
        self.model = NORESQA(output=40, output2=40, metric_type=self.metric_type, config_path=str(CONFIG_PATH.resolve()))

        # load model
        pretrained_dict = {}
        for k, v in state.items():
            if "module" in k:
                pretrained_dict[k.replace("module.", "")] = v
            else:
                pretrained_dict[k] = v
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(pretrained_dict)

        # change device as needed
        # device
        if self.gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(self.gpu_id))
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

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
                    score = self.__calculate_noresqamos(hyp_file)

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

    def __calculate_noresqamos(self, degraded_file_path: str):
        try:
            nmr_feat, test_feat = feats_loading(degraded_file_path, self.nmr, noresqa_or_noresqaMOS=self.metric_type)
            test_feat = torch.from_numpy(test_feat).float().to(self.device).unsqueeze(0)
            nmr_feat = torch.from_numpy(nmr_feat).float().to(self.device).unsqueeze(0)

            with torch.no_grad():
                mos_score = self.model(nmr_feat, test_feat).detach().cpu().numpy()[0]

            return 5.0 - mos_score  # 5.0 is the best score and 1.0 is the worst score
        except Exception as e:
            raise Exception(f"Error while calculating NISQA score: {e}")


if __name__ == "__main__":
    metric = NoresqaMos(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([metric])
