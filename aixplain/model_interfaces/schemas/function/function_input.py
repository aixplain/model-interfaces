from enum import Enum
from http import HTTPStatus
from typing import Optional, Any, List, Union, Tuple
import numpy as np

from pydantic import BaseModel, validator, Extra
import tornado.web

from aixplain.model_interfaces.utils import serialize
from aixplain.model_interfaces.schemas.api.basic_api_input import APIInput
from aixplain.model_interfaces.schemas.modality.modality_input import TextInput

class AudioEncoding(Enum):
    """
    All supported audio encoding formats by the interface are
    enlisted in this enum. 
    """
    
    WAV = "wav" # 16 bit Linear PCM default wav format

class AudioConfig(BaseModel):
    """The standardized schema of the aiXplain's audio serialized data config.
    :param audio_encoding:
        Audio format of the audio data before base64 encoding.
        Chosen from the supported types in the enum AudioEncoding
        Supported encoding(s): wav
    :type audio_encoding:
        AudioEncoding
    :param sampling_rate:
        Sampling rate in hertz for the audio data. Optional.
    :type sampling_rate:
        int
    """
    audio_encoding: AudioEncoding
    sampling_rate: Optional[int]

class SpeechRecognitionInputSchema(APIInput):
    """The standardized schema of the aiXplain's Speech Recognition API input.
    
    :param data:
        Input data to the model.
        Serialized base 64 encoded audio data in the audio encoding defined
        by the audio_config parameter.
    :type data:
        str
    :param audio_config:
        Configuration specifying the audio encoding parameters of the provided
        input.
    :type audio_config:
        AudioConfig
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. 
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.
    :type version:
        str
    :param language:
        The source language the model processes for Speech Recognition.
    :type language:
        str
    :param dialect:
        The source dialect the model processes (if specified) for Speech Recognition.
        Optional.
    :type dialect:
        str
    """
    data: str
    audio_config: AudioConfig
    language: str
    dialect: Optional[str] = ""

    @validator('data')
    def decode_data(cls, v):
        decoded = serialize.decode(v)
        return decoded

class SpeechRecognitionInput(SpeechRecognitionInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into SpeechRecognitionInput."
                )

class ClassificationInputSchema(APIInput):
    """The standardized schema of the aiXplain's classification API input.
    
    :param data:
        Input data to the model.
    :type data:
        Any
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. 
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.
    :type version:
        str
    :param language:
        The source language the model processes for classification.
    :type language:
        str
    :param dialect:
        The source dialect the model processes (if specified) for classification.
        Optional.
    :type dialect:
        str
    """
    language: Optional[str] = ""
    dialect: Optional[str] = ""

class ClassificationInput(ClassificationInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into DiarizationInput."
                )

class SpeechEnhancementInputSchema(APIInput):
    """The standardized schema of the aiXplain's Speech Enhancement API input.
    
    :param data:
        Input data to the model.
        Serialized base 64 encoded audio data in the audio encoding defined
        by the audio_config parameter.
    :type data:
        str
    :param audio_config:
        Configuration specifying the audio encoding parameters of the provided
        input.
    :type audio_config:
        AudioConfig
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. 
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.
    :type version:
        str
    :param language:
        The source language the model processes for Speech Recognition.
    :type language:
        str
    :param dialect:
        The source dialect the model processes (if specified) for Speech Recognition.
        Optional.
    :type dialect:
        str
    """
    data: str
    audio_config: AudioConfig
    language: str
    dialect: Optional[str] = ""

    @validator('data')
    def decode_data(cls, v):
        decoded = serialize.decode(v)
        return decoded

class SpeechEnhancementInput(SpeechEnhancementInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into SpeechRecognitionInput."
                )

class SpeechSynthesisInputSchema(BaseModel):
    """The standardized schema of the aiXplain's Speech Synthesis input.
    
    :param audio:
        Input audio to the model.
        Serialized base 64 encoded audio data in the audio encoding defined
        by the audio_config parameter.
        or a link to the audio file
    :type audio:
        str
    :param text:
        Input text to synthesize into audio.
    :type text:
        str
    :param language:
        Supplier name.
    :type language:
        str
    :param audio_config:
        Configuration specifying the audio encoding parameters of the provided
    :type audio_config:
        AudioConfig
    """
    speaker_id: str
    data: Optional[str] = ""
    audio: str = ""
    text: str
    text_language: str = "en"
    audio_config: AudioConfig

class SpeechSynthesisInput(SpeechSynthesisInputSchema):
    def __init__(self, **input):
        data = input.get('data')
        if data == "":
            super().__init__(**input)
            try:
                super().__init__(**input)
            except ValueError:
                raise tornado.web.HTTPError(
                        status_code=HTTPStatus.BAD_REQUEST,
                        reason="Incorrect types passed into SpeechSynthesisInput."
                    )
        else:
            raise tornado.web.HTTPError(
                        status_code=HTTPStatus.BAD_REQUEST,
                        reason="Incorrect types passed into SpeechSynthesisInput. data field shouldn't be pass, and the data for audio file should be passed in field with the name [audio]"
                    )

class TextToImageGenerationInputSchema(BaseModel):
    """The standardized schema of the aiXplain's Text-based Image Generation API input.
    
    :param data:
        Input data to the model.
        Prompt for image generation
    :type data:
        str
    """
    data: str

class TextToImageGenerationInput(TextToImageGenerationInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect type passed into TextToImageGenerationInput."
                )

class Segment(BaseModel):
    """Segment information with optional url field. If the url field populated
    this means the segmentation performed and the regarding segment uploaded
    to the defined location under `url` field.
    """
    segment_id: int
    start: Union[float, int, Tuple[int, int]]
    end: Union[float, int, Tuple[int, int]]
    url: Optional[str]

class SegmentationInputSchema(APIInput):
    """The standardized schema of the aiXplain's Segmenation API input.

    :param details:
        List of segments in Segment type.
    :type data:
        Segment
    """
    details: Optional[Union[List[Segment], str]]

class SegmentationInput(SegmentationInputSchema):
    pass
        
class TextGenerationInputSchema(TextInput):
    """The standardized schema of aiXplains text generation API Input


    """
    temperature: Optional[float] = 1.0
    max_new_tokens: Optional[int] = 200
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 40
    num_return_sequences: Optional[int] = 1
    script: Optional[str] = ""

class TextGenerationInput(TextGenerationInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect type passed into TextGenerationInput."
                )
        

class TranslationInputSchema(TextInput):
    """The standardized schema of the aiXplain's Translation API input.
    
    :param data:
        Input data to the model.
    :type data:
        Any
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. 
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.  
    :type version:
        str
    :param source_language:
        The source language the model processes for translation.
    :type source_language:
        str
    :param source_dialect:
        The source dialect the model processes (if specified) for translation.
        Optional.
    :type source_dialect:
        str
    :param target_language:
        The target language the model processes for translation.
    :type target_language:
        str
    """
    source_language: str
    source_dialect: Optional[str] = ""
    target_language: str
    target_dialect: Optional[str] = ""

class TranslationInput(TranslationInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into TranslationInput."
                )
        
class TextSummarizationInputSchema(TextInput):
    """The standardized schema of the aiXplain's text summarization API input.

    :param data:
        Input data to the model.
    :type data:
        str
    :param supplier:
        Supplier name. Optional.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. Optional.
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.
    :param language:
        The model's language. Optional.
    :type language:
        str
    :param script:
        # TODO What is this?
    :type script:
        str
    :param dialect:
        The language's dialect. Optional.
    :type dialect:
        str 
    """
    language: Optional[str] = ""
    script: Optional[str] = ""
    dialect: Optional[str] = ""

class TextSummarizationInput(TextSummarizationInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into TextSummarizationInput."
                )

class SearchInputSchema(TextInput):
    """The standardized schema of the aiXplain's text summarization API input.

    :param data:
        Input data to the model.
    :type data:
        str
    :param supplier:
        Supplier name. Optional.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. Optional.
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.
    :param language:
        The model's language. Optional.
    :type language:
        str
    :param script:
        # TODO What is this?
    :type script:
        str
    :param supplier_model_id:
        The model ID from the supplier. Optional.
    :type supplier_model_id:
        str
    """
    script: Optional[str] = ""
    supplier_model_id: Optional[str] = ""

class SearchInput(SearchInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into SearchInput."
                )
        
class DiacritizationInputSchema(TextInput):
    """The standardized schema of the aiXplain's diacritization API input.
    
    :param data:
        Input data to the model.
    :type data:
        Any
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. 
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.
    :type version:
        str
    :param language:
        The source language the model processes for diarization.
    :type language:
        str
    :param dialect:
        The source dialect the model processes (if specified) for diarization.
        Optional.
    :type dialect:
        str
    """
    dialect: Optional[str] = ""

class DiacritizationInput(DiacritizationInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into DiacritizationInput."
                )

class TextReconstructionInputSchema(TextInput):
    """The standardized schema of the aiXplain's text reconstruction API input.
    
    :param data:
        Input data to the model.
    :type data:
        Any
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. 
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.
    :type version:
        str
    :param language:
        The source language the model processes for diarization.
    :type language:
        str
    """
    pass

class TextReconstructionInput(TextReconstructionInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into TextReconstructionInput."
                )
        
class FillTextMaskInputSchema(TextInput):
    """The standardized schema of the aiXplain's fill-text-mask API input.
    
    :param data:
        Input data to the model.
    :type data:
        Any
    :param supplier:
        Supplier name. Optional.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. Optional.
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.
    :type version:
        str
    :param language:
        The source language the model processes for diarization.
    :type language:
        str
    :param dialect:
        The source dialect the model processes (if specified) for diarization.
        Optional.
    :type dialect:
        str
    :param script:
        # TODO What is this? Optional.
    :type script:
        str
    """
    language: str
    dialect: Optional[str]
    script: Optional[str]

class FillTextMaskInput(FillTextMaskInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into FillTextMaskInput."
                )

class SubtitleTranslationInputSchema(TextInput):
    """The standardized schema of the aiXplain's subtitle translation API input.
    
    :param data:
        Input data to the model.
    :type data:
        Any
    :param supplier:
        Supplier name. Optional.
    :type supplier:
        str
    :param function:
        The aixplain function name for the model. Optional.
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.
    :type version:
        str
    :param source_language:
        The subtitle's source language.
    :type source_language:
        str
    :param dialect_in:
        The dialect of the source language. Optional.
    :type dialect_in:
        str
    :param target_supplier:
        TODO What is this?
    :type target_supplier:
        str
    :param target_languages:
        Languages to which to translate the subtitle.
    :type target_languages:
        List[str]
    """
    source_language: str
    dialect_in: Optional[str]
    target_supplier: Optional[str]
    target_languages: Optional[List[str]]

class SubtitleTranslationInput(SubtitleTranslationInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect types passed into SubtitleTranslationInput."
                )

class AutoMaskGenerationInputSchema(TextInput, extra=Extra.allow):
    """The standardized schema of the aiXplain's automatic image masking function.
    Note that most of these fields are catered specifically toward SAM2 and do 
    not all all have to be implemented. Documentation from 
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py

    :param data:
        Input data to the model.
    :type data:
        Text
    :param points_per_side:
        The number of points to be sampled along one side of the image. The 
        total number of points is points_per_side**2. If None, 'point_grids' 
        must provide explicit point sampling.
    :type points_per_side:
        int
    :param points_per_batch:
        Sets the number of points run simultaneously by the model. Higher 
        numbers may be faster but use more GPU memory.
    :type points_per_batch:
        int
    :param pred_iou_thresh:
        A filtering threshold in [0,1], using the model's predicted mask 
        quality.
    :type pred_iou_thresh:
        float
    :param stability_score_thresh:
        A filtering threshold in [0,1], using the stability of the mask under 
        changes to the cutoff used to binarize the model's mask predictions.
    :type stability_score_thresh:
        float
    :param stability_score_offset:
        The amount to shift the cutoff when calculated the stability score.
    :type stability_score_offset:
        float
    :param box_nms_thresh:
        The box IoU cutoff used by non-maximal suppression to filter duplicate 
        masks.
    :type box_nms_thresh:
        float
    :param crop_n_layers:
        If >0, mask prediction will be run again on crops of the image. Sets 
        the number of layers to run, where each layer has 2**i_layer number of 
        image crops.
    :type crop_n_layers:
        int
    :param crop_nms_thresh:
        The box IoU cutoff used by non-maximal suppression to filter duplicate 
        masks between different crops.
    :type crop_nms_thresh:
        float
    :param crop_overlap_ratio:
        Sets the degree to which crops overlap. In the first crop layer, crops 
        will overlap by this fraction of the image length. Later layers with 
        more crops scale down this overlap.
    :type crop_overlap_ratio:
        float
    :param crop_n_points_downscale_factor:
        The number of points-per-side sampled in layer n is scaled down by 
        crop_n_points_downscale_factor**n.
    :type crop_n_points_downscale_factor:
        int
    :param point_grids:
        A list over explicit grids of points used for sampling, normalized to 
        [0,1]. The nth grid in the list is used in the nth crop layer. 
        Exclusive with points_per_side.
    :type point_grids:
        List[ndarray]
    :param min_mask_region_area:
        If >0, postprocessing will be applied to remove disconnected regions 
        and holes in masks with area smaller than min_mask_region_area. 
        Requires opencv.
    :type min_mask_region_area:
        int
    :param output_mode:
        The form masks are returned in. Can be 'binary_mask', 
        'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools. For 
        large resolutions, 'binary_mask' may consume large amounts of memory.
    :type output_mode:
        str
    """
    points_per_side: Optional[int] = 32
    points_per_batch: Optional[int] = 64
    pred_iou_thresh: Optional[float] = 0.88
    stability_score_thresh: Optional[float] = 0.95
    stability_score_offset: Optional[float] = 1.0
    box_nms_thresh: Optional[float] = 0.7
    crop_n_layers: Optional[int] = 0
    crop_nms_thresh: Optional[float] = 0.7
    crop_overlap_ratio: Optional[float] = 512 / 1500
    crop_n_points_downscale_factor: Optional[int] = 1
    point_grids: Optional[List[List]] = None
    min_mask_region_area: Optional[int] = 0
    output_mode: Optional[str] = "binary_mask"

class AutoMaskGenerationInput(AutoMaskGenerationInputSchema):
    def __init__(self, **input):
        super().__init__(**input)
        try:
            super().__init__(**input)
        except ValueError:
            raise tornado.web.HTTPError(
                    status_code=HTTPStatus.BAD_REQUEST,
                    reason="Incorrect type passed into TextGenerationInput."
                )