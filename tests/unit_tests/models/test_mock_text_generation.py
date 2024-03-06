from unittest.mock import Mock
from typing import Dict, List

from aixplain.model_interfaces.schemas.modality.modality_input import TextInput
from aixplain.model_interfaces.schemas.function.function_input import TextGenerationInput
from aixplain.model_interfaces.schemas.function.function_output import TextGenerationOutput
from aixplain.model_interfaces.interfaces.function_models import TextGenerationModel
from aixplain.model_interfaces.interfaces.function_models import TextGenerationChatModel

class TestMockTextGeneration():
    def test_predict(self):
        data = "Hello, how are you?"
        supplier = "mockGpt"
        function = "text-generation"
        version = ""
        language = ""

        text_generation_input_dict = {
            "data": data,
            "supplier": supplier,
            "function": function,
            "version": version,
            "language": language
        }
        predict_input = {"instances": [text_generation_input_dict]}
        
        mock_model = MockModel("Mock")
        predict_output = mock_model.predict(predict_input)
        text_generation_output_dict = predict_output["predictions"][0]

        assert text_generation_output_dict["data"] == "I am a text generation model."

    def test_count_tokens(self):
        messages = messages = [TextInput(**{"data": "mock"}) for _ in range(20)]
        mock_model = MockModel("Mock")
        tokenize_output = mock_model.count_tokens(messages)
        assert tokenize_output == [5 for _ in range(20)]

class TestMockTextGenerationChat():
    def test_templatize(self):
        data = "Hello, how are you?"
        supplier = "mockGpt"
        function = "text-generation"
        version = ""
        language = ""
        prompt = "mock prompt"
        context = "mock context"
        history = {}

        text_generation_input_dict = {
            "data": data,
            "supplier": supplier,
            "function": function,
            "version": version,
            "language": language,
            "prompt": prompt,
            "context": context,
            "history": history
        }
        predict_input = {"instances": [text_generation_input_dict]}
        
        mock_model = MockChatModel("Mock")
        templatize_output = mock_model.templatize([predict_input])
        for output in templatize_output:
            assert dict(output)["data"] == "mock prompt input"
        

    def test_predict(self):
        data = "Hello, how are you?"

        text_generation_input_dict = {
            "data": data,
        }
        predict_input = {"instances": [text_generation_input_dict]}
        
        mock_model = MockChatModel("Mock")
        predict_output = mock_model.predict(predict_input)
        text_generation_output_dict = predict_output["predictions"][0]

        assert text_generation_output_dict["data"] == "I am a text generation model."

    def test_count_tokens(self):
        messages = [TextInput(**{"data": "mock"}) for _ in range(20)]
        mock_model = MockModel("Mock")
        tokenize_output = mock_model.count_tokens(messages)
        assert tokenize_output == [5 for _ in range(20)]

class MockModel(TextGenerationModel):
    def run_model(self, api_input: Dict[str, List[TextInput]], headers: Dict[str, str] = None) -> Dict[str, List[TextGenerationOutput]]:
        instances = api_input["instances"]
        predictions_list = []
        # There's only 1 instance in this case.
        for instance in instances:
            instance_data = instance.dict()
            model_instance = Mock()
            model_instance.process_data.return_value = "I am a text generation model."
            result = model_instance.process_data(instance_data["data"])
            model_instance.delete()
            
            # Map back onto TextGenerationOutput
            data = result

            output_dict = {
                "data": data,
            }
            text_generation_output = TextGenerationOutput(**output_dict)
            predictions_list.append(text_generation_output)
        predict_output = {"predictions": predictions_list}
        return predict_output

    def count_tokens(self, messages: List[TextInput]) -> List[int]:
        return [5 for _ in messages]
    
class MockChatModel(TextGenerationChatModel):
    def run_model(self, api_input: Dict[str, List[TextInput]], headers: Dict[str, str] = None) -> Dict[str, List[TextGenerationOutput]]:
        instances = api_input["instances"]
        predictions_list = []
        # There's only 1 instance in this case.
        for instance in instances:
            instance_data = instance.dict()
            model_instance = Mock()
            model_instance.process_data.return_value = "I am a text generation model."
            result = model_instance.process_data(instance_data["data"])
            model_instance.delete()
            
            # Map back onto TextGenerationOutput
            data = result

            output_dict = {
                "data": data,
            }
            text_generation_output = TextGenerationOutput(**output_dict)
            predictions_list.append(text_generation_output)
        predict_output = {"predictions": predictions_list}
        return predict_output

    def count_tokens(self, messages: List[TextInput]) -> List[int]:
        return [5 for _ in messages]
    
    def templatize(self, inputs: List[TextGenerationInput]) -> List[TextInput]:
        ret_list = []
        for _ in inputs:
            ret_list.append(TextInput(**{"data": "mock prompt input"}))
        return ret_list