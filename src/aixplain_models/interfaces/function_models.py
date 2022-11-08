from http import HTTPStatus
from typing import Dict, List

from aixplain_models.schemas.function_input import (
    TranslationInput,
    SpeechRecognitionInput,
    DiacritizationInput,
    ClassificationInput,
    SpeechEnhancementInput
)
from aixplain_models.schemas.function_output import (
    TranslationOutput,
    SpeechRecognitionOutput,
    DiacritizationOutput,
    ClassificationOutput,
    SpeechEnhancementOutput
)
from aixplain_models.interfaces.aixplain_model import AixplainModel

class TranslationModel(AixplainModel):
    def run_model(self, api_input: Dict[str, List[TranslationInput]]) -> Dict[str, List[TranslationOutput]]:
        pass 

    def predict(self, request: Dict) -> Dict:
        instances = request['instances']
        translation_input_list = []
        # Convert JSON serializables into TranslationInputs
        for instance in instances:
            translation_input = TranslationInput(**instance)
            translation_input_list.append(translation_input)
        translation_output = self.run_model({"instances": translation_input_list})

        # Convert JSON serializables into TranslationOutputs
        for i in range(len(translation_output["predictions"])):
            translation_output_dict = translation_output["predictions"][i].dict()
            TranslationOutput(**translation_output_dict)
            translation_output["predictions"][i] = translation_output_dict
        return translation_output

class SpeechRecognitionModel(AixplainModel):
    def run_model(self, api_input: Dict[str, List[SpeechRecognitionInput]]) -> Dict[str, List[SpeechRecognitionOutput]]:
        pass 

    def predict(self, request: Dict) -> Dict:
        instances = request['instances']
        sr_input_list = []
        # Convert JSON serializables into SpeechRecognitionInputs
        for instance in instances:
            sr_input = SpeechRecognitionInput(**instance)
            sr_input_list.append(sr_input)
        sr_output = self.run_model({"instances": sr_input_list})

        # Convert JSON serializables into SpeechRecognitionOutputs
        for i in range(len(sr_output["predictions"])):
            sr_output_dict = sr_output["predictions"][i].dict()
            SpeechRecognitionOutput(**sr_output_dict)
            sr_output["predictions"][i] = sr_output_dict
        return sr_output

class DiacritizationModel(AixplainModel):
    def run_model(self, api_input: Dict[str, List[DiacritizationInput]]) -> Dict[str, List[DiacritizationOutput]]:
        pass 

    def predict(self, request: Dict) -> Dict:
        instances = request['instances']
        diacritiztn_input_list = []
        # Convert JSON serializables into DiacritizationInputs
        for instance in instances:
            diacritiztn_input = DiacritizationInput(**instance)
            diacritiztn_input_list.append(diacritiztn_input)
        diacritiztn_output = self.run_model({"instances": diacritiztn_input_list})

        # Convert JSON serializables into DiacritizationOutputs
        for i in range(len(diacritiztn_output["predictions"])):
            diacritiztn_output_dict = diacritiztn_output["predictions"][i].dict()
            DiacritizationOutput(**diacritiztn_output_dict)
            diacritiztn_output["predictions"][i] = diacritiztn_output_dict
        return diacritiztn_output

class ClassificationModel(AixplainModel):
    def run_model(self, api_input: Dict[str, List[ClassificationInput]]) -> Dict[str, List[ClassificationOutput]]:
        pass 

    def predict(self, request: Dict) -> Dict:
        instances = request['instances']
        classification_input_list = []
        # Convert JSON serializables into ClassificationInputs
        for instance in instances:
            classification_input = ClassificationInput(**instance)
            classification_input_list.append(classification_input)
        classification_output = self.run_model({"instances": classification_input_list})

        # Convert JSON serializables into ClassificationOutputs
        for i in range(len(classification_output["predictions"])):
            classification_output_dict = classification_output["predictions"][i].dict()
            ClassificationOutput(**classification_output_dict)
            classification_output["predictions"][i] = classification_output_dict
        return classification_output

class SpeechEnhancementModel(AixplainModel):
    def run_model(self, api_input: Dict[str, List[SpeechEnhancementInput]]) -> Dict[str, List[SpeechEnhancementOutput]]:
        pass 

    def predict(self, request: Dict) -> Dict:
        instances = request['instances']
        speech_enhancement_input_list = []
        # Convert JSON serializables into SpeechEnhancementInputs
        for instance in instances:
            speech_enhancement_input = SpeechEnhancementInput(**instance)
            speech_enhancement_input_list.append(speech_enhancement_input)
        speech_enhancement_output = self.run_model({"instances": speech_enhancement_input_list})

        # Convert JSON serializables into SpeechEnhancementOutputs
        for i in range(len(speech_enhancement_output["predictions"])):
            speech_enhancement_dict = speech_enhancement_output["predictions"][i].dict()
            SpeechEnhancementOutput(**speech_enhancement_dict)
            try:
                speech_enhancement_dict["audio_config"]["audio_encoding"] = speech_enhancement_dict["audio_config"]["audio_encoding"].value
            except AttributeError as e:
                raise tornado.web.HTTPError(
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                    reason="The user request, although correct, is generating unacceptable output from the server."
                )
            
            speech_enhancement_output["predictions"][i] = speech_enhancement_dict
        return speech_enhancement_output