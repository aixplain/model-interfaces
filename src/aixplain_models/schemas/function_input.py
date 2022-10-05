from typing import Optional, Any
import tornado
from http import HTTPStatus
from pydantic import BaseModel

class APIInput(BaseModel):
    """The standardized schema of the aiXplain's API input.
    
    :param data:
        Input data to the supplier model.
    :type data:
        Any
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The functionality of the supplier's model. 
    :type function:
        str 
    :param version:
        The version number of the model if the supplier has multiple 
        models with the same function. Optional.  
    :type version:
        str
    :param language:
        The language the model processes (if relevant). Optional.
    :type language:
        str
    """
    data: Any
    supplier: str
    function: str
    version: Optional[str] = ""
    language: Optional[str] = ""

class TranslationInputSchema(APIInput):
    """The standardized schema of the aiXplain's Translation API input.
    
    :param data:
        Input data to the supplier model.
    :type data:
        Any
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The functionality of the supplier's model. 
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

class SpeechRecognitionInputSchema(APIInput):
    """The standardized schema of the aiXplain's Speech Recognition API input.
    
    :param data:
        Input data to the supplier model.
    :type data:
        Any
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The functionality of the supplier's model. 
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
    language: str
    dialect: Optional[str] = ""

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

class DiacritizationInputSchema(APIInput):
    """The standardized schema of the aiXplain's diacritization API input.
    
    :param data:
        Input data to the supplier model.
    :type data:
        Any
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The functionality of the supplier's model. 
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
    language: str
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

class ClassificationInputSchema(APIInput):
    """The standardized schema of the aiXplain's classification API input.
    
    :param data:
        Input data to the supplier model.
    :type data:
        Any
    :param supplier:
        Supplier name.
    :type supplier:
        str
    :param function:
        The functionality of the supplier's model. 
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