import tornado.web

from http import HTTPStatus
from typing import Dict, List

from aixplain.model_interfaces.schemas.function.function_input import (
    TranslationInput,
    SpeechRecognitionInput,
    DiacritizationInput,
    ClassificationInput,
    SpeechEnhancementInput,
    SpeechSynthesisInput,
    TextToImageGenerationInput,
    TextGenerationInput,
    TextSummarizationInput,
    SearchInput
)
from aixplain.model_interfaces.schemas.function.function_output import (
    TranslationOutput,
    SpeechRecognitionOutput,
    DiacritizationOutput,
    ClassificationOutput,
    SpeechEnhancementOutput,
    SpeechSynthesisOutput,
    TextToImageGenerationOutput,
    TextGenerationOutput,
    TextSummarizationOutput,
    SearchOutput
)
from aixplain.model_interfaces.interfaces.aixplain_model import AixplainModel

class TranslationModel(AixplainModel):

    def run_model(self, api_input: Dict[str, List[TranslationInput]], headers: Dict[str, str] = None) -> Dict[str, List[TranslationOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:
        instances = request['instances']
        translation_input_list = []
        # Convert JSON serializables into TranslationInputs
        for instance in instances:
            translation_input = TranslationInput(**instance)
            translation_input_list.append(translation_input)
        translation_output = self.run_model({"instances": translation_input_list}, headers)

        # Convert JSON serializables into TranslationOutputs
        for i in range(len(translation_output["predictions"])):
            translation_output_dict = translation_output["predictions"][i].dict()
            TranslationOutput(**translation_output_dict)
            translation_output["predictions"][i] = translation_output_dict
        return translation_output

class SpeechRecognitionModel(AixplainModel):

    def run_model(self, api_input: Dict[str, List[SpeechRecognitionInput]], headers: Dict[str, str] = None) -> Dict[str, List[SpeechRecognitionOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:

        instances = request['instances']
        sr_input_list = []
        # Convert JSON serializables into SpeechRecognitionInputs
        for instance in instances:
            sr_input = SpeechRecognitionInput(**instance)
            sr_input_list.append(sr_input)
        sr_output = self.run_model({"instances": sr_input_list}, headers)


        # Convert JSON serializables into SpeechRecognitionOutputs
        for i in range(len(sr_output["predictions"])):
            sr_output_dict = sr_output["predictions"][i].dict()
            SpeechRecognitionOutput(**sr_output_dict)
            sr_output["predictions"][i] = sr_output_dict
        return sr_output

class DiacritizationModel(AixplainModel):

    def run_model(self, api_input: Dict[str, List[DiacritizationInput]], headers: Dict[str, str] = None) -> Dict[str, List[DiacritizationOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:
        instances = request['instances']
        diacritiztn_input_list = []
        # Convert JSON serializables into DiacritizationInputs
        for instance in instances:
            diacritiztn_input = DiacritizationInput(**instance)
            diacritiztn_input_list.append(diacritiztn_input)
        diacritiztn_output = self.run_model({"instances": diacritiztn_input_list}, headers)

        # Convert JSON serializables into DiacritizationOutputs
        for i in range(len(diacritiztn_output["predictions"])):
            diacritiztn_output_dict = diacritiztn_output["predictions"][i].dict()
            DiacritizationOutput(**diacritiztn_output_dict)
            diacritiztn_output["predictions"][i] = diacritiztn_output_dict
        return diacritiztn_output

class ClassificationModel(AixplainModel):
    def run_model(self, api_input: Dict[str, List[ClassificationInput]], headers: Dict[str, str] = None) -> Dict[str, List[ClassificationOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:
        instances = request['instances']
        classification_input_list = []
        # Convert JSON serializables into ClassificationInputs
        for instance in instances:
            classification_input = ClassificationInput(**instance)
            classification_input_list.append(classification_input)
        classification_output = self.run_model({"instances": classification_input_list}, headers)

        # Convert JSON serializables into ClassificationOutputs
        for i in range(len(classification_output["predictions"])):
            classification_output_dict = classification_output["predictions"][i].dict()
            ClassificationOutput(**classification_output_dict)
            classification_output["predictions"][i] = classification_output_dict
        return classification_output

class SpeechEnhancementModel(AixplainModel):

    def run_model(self, api_input: Dict[str, List[SpeechEnhancementInput]], headers: Dict[str, str] = None) -> Dict[str, List[SpeechEnhancementOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:
        instances = request['instances']
        speech_enhancement_input_list = []
        # Convert JSON serializables into SpeechEnhancementInputs
        for instance in instances:
            speech_enhancement_input = SpeechEnhancementInput(**instance)
            speech_enhancement_input_list.append(speech_enhancement_input)
        speech_enhancement_output = self.run_model({"instances": speech_enhancement_input_list}, headers)

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


class SpeechSynthesis(AixplainModel):
    def run_model(self, api_input: Dict[str, List[SpeechSynthesisInput]], headers: Dict[str, str] = None) -> Dict[str, List[SpeechSynthesisOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:
        instances = request['instances']
        speech_synthesis_input_list = []
        # Convert JSON serializables into SpeechEnhancementInputs
        for instance in instances:
            speech_synthesis_input = SpeechSynthesisInput(**instance)
            speech_synthesis_input_list.append(speech_synthesis_input)

        speech_synthesis_output = self.run_model({"instances": speech_synthesis_input_list}, headers)


        # Convert JSON serializables into SpeechEnhancementOutputs
        for i in range(len(speech_synthesis_output["instances"])):
            speech_synthesis_dict = speech_synthesis_output["instances"][i].dict()
            SpeechSynthesisOutput(**speech_synthesis_dict)
            speech_synthesis_output["instances"][i] = speech_synthesis_dict
        return speech_synthesis_output

class TextToImageGeneration(AixplainModel):
    def run_model(self, api_input: Dict[str, List[TextToImageGenerationInput]], headers: Dict[str, str] = None) -> Dict[str, List[TextToImageGenerationOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:
        instances = request['instances']
        text_to_image_generation_input_list = []
        # Convert JSON serializables into TextToImageGenerationInputs
        for instance in instances:
            text_to_image_generation_input = TextToImageGenerationInput(**instance)
            text_to_image_generation_input_list.append(text_to_image_generation_input)
            
        text_to_image_generation_output = self.run_model({"instances": text_to_image_generation_input_list})

        # Convert JSON serializables into TextToImageGenerationOutputs
        for i in range(len(text_to_image_generation_output["predictions"])):
            text_to_image_generation_dict = text_to_image_generation_output["predictions"][i].dict()
            TextToImageGenerationOutput(**text_to_image_generation_dict)
            text_to_image_generation_output["predictions"][i] = text_to_image_generation_dict
        return text_to_image_generation_output
    
class TextGenerationModel(AixplainModel):
    def run_model(self, api_input: Dict[str, List[TextGenerationInput]], headers: Dict[str, str] = None) -> Dict[str, List[TextGenerationOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:
        instances = request['instances']
        text_generation_input_list = []
        # Convert JSON serializables into TextGenerationInputs
        for instance in instances:
            text_generation_input = TextGenerationInput(**instance)
            text_generation_input_list.append(text_generation_input)
            
        text_generation_output = self.run_model({"instances": text_generation_input_list})

        # Convert JSON serializables into TextGenerationOutputs
        for i in range(len(text_generation_output["predictions"])):
            text_generation_dict = text_generation_output["predictions"][i].dict()
            TextGenerationOutput(**text_generation_dict)
            text_generation_output["predictions"][i] = text_generation_dict
        return text_generation_output

class TextSummarizationModel(AixplainModel):
    def run_model(self, api_input: Dict[str, List[TextSummarizationInput]], headers: Dict[str, str] = None) -> Dict[str, List[TextSummarizationOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:
        instances = request['instances']
        text_summarization_input_list = []
        # Convert JSON serializables into TextSummarizationInputs
        for instance in instances:
            text_summarization_input = TextSummarizationInput(**instance)
            text_summarization_input_list.append(text_summarization_input)
            
        text_summarization_output = self.run_model({"instances": text_summarization_input_list})

        # Convert JSON serializables into TextSummarizationOutputs
        for i in range(len(text_summarization_output["predictions"])):
            text_summarization_dict = text_summarization_output["predictions"][i].dict()
            TextSummarizationOutput(**text_summarization_dict)
            text_summarization_output["predictions"][i] = text_summarization_dict
        return text_summarization_output
    
class SearchModel(AixplainModel):
    def run_model(self, api_input: Dict[str, List[SearchInput]], headers: Dict[str, str] = None) -> Dict[str, List[SearchOutput]]:
        pass

    def predict(self, request: Dict[str, str], headers: Dict[str, str] = None) -> Dict:
        instances = request['instances']
        search_input_list = []
        # Convert JSON serializables into SearchInputs
        for instance in instances:
            search_input = SearchInput(**instance)
            search_input_list.append(search_input)
            
        search_output = self.run_model({"instances": search_input_list})

        # Convert JSON serializables into SearchOutputs
        for i in range(len(search_output["predictions"])):
            search_dict = search_output["predictions"][i].dict()
            SearchOutput(**search_dict)
            search_output["predictions"][i] = search_dict
        return search_output
    