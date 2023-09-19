__author__ = "aiXplain"

import numpy as np
import os

from aixplain.model_schemas.interfaces.aixplain_model_server import AixplainModelServer
from aixplain.model_schemas.interfaces.metric_models import ReferencelessTextGenerationMetric
from aixplain.model_schemas.interfaces.asset_resolver import AssetResolver
from aixplain.model_schemas.schemas.metric_input import ReferencelessTextGenerationMetricInput
from aixplain.model_schemas.schemas.metric_output import ReferencelessTextGenerationMetricOutput
from typing import Dict, List
from .external.norefer import NoRefER
from pathlib import Path


MODEL_NOT_FOUND_ERROR = """
    Download model file using command:
    # TODO (krishnadurai): Host this on a public URL
    aws s3 cp --recursive s3://benchmarksdata/models/norefer/ ./external/
"""


class NoRefERMetric(ReferencelessTextGenerationMetric):
    def load(self):
        model_path = Path(AssetResolver.resolve_path()) / "epoch_1.ckpt"

        if not os.path.exists(model_path):
            raise ValueError(MODEL_NOT_FOUND_ERROR)

        self.model = NoRefER("nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large")
        self.model.load_from_checkpoint(model_path)
        self.ready = True

    def run_metric(self, request: Dict[str, List[ReferencelessTextGenerationMetricInput]]) -> Dict[str, List[ReferencelessTextGenerationMetricInput]]:
        """Scoring Function for NoRef-ER Metric

        Args:
            request (Dict[str, List[ReferencelessTextGenerationMetricInput]]): input request with `instances` key

        Returns:
            Dict[str, List[ReferencelessTextGenerationMetricInput]]: output
        """
        inputs = request["instances"]

        predictions = []
        for inp in inputs:
            hypotheses, sources = inp.hypotheses, inp.sources

            seg_scores = []
            for hyp, src in zip(hypotheses, sources):
                seg_scores.append(self.model.get_score(hyp))

            sys_score = round(np.average(seg_scores), 2)

            return_dict = {"corpus-level": round(sys_score, 4), "sentence-level": [round(s, 4) for s in seg_scores]}

            output_dict = ReferencelessTextGenerationMetricOutput(**{"data": return_dict["corpus-level"], "details": return_dict})
            predictions.append(output_dict)

        predict_output = {"scores": predictions}
        return predict_output


if __name__ == "__main__":
    metric = NoRefERMetric(AssetResolver.asset_uri())
    AixplainModelServer(workers=1).start([metric])
