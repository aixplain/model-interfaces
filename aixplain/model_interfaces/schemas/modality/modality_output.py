"""
Modality output classes for modality-based model classification
"""
from http import HTTPStatus
from typing import Optional, Any

from aixplain.model_interfaces.schemas.api.basic_api_output import APIOutput

class TextOutput(APIOutput):
    """The standardized schema of the aiXplain's text API outputs.

    :param data:
        Input data to the model.
    :type data:
        str
    """
    data: str