from unittest.mock import Mock
from aixplain.model_interfaces.schemas.function.function_input import TextGenerationInput
from aixplain.model_interfaces.schemas.function.function_output import TextGenerationOutput
from aixplain.model_interfaces.interfaces.function_models import TextGenerationModel
from typing import Dict, List

class TestMockTextGeneration():
    def test_predict(self):
        data = "Hello, how are you?"
        supplier = "mockGpt"
        function = "text-generation"
        version = ""
        language = ""
        temperature = 1.0
        max_new_tokens = 200
        top_p = 0.8
        top_k = 40
        num_return_sequences = 1

        text_generation_input_dict = {
            "data": data,
            "supplier": supplier,
            "function": function,
            "version": version,
            "language": language,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences
        }
        predict_input = {"instances": [text_generation_input_dict]}
        
        mock_model = MockModel("Mock")
        predict_output = mock_model.predict(predict_input)
        text_generation_output_dict = predict_output["predictions"][0]

        assert text_generation_output_dict["data"] == "I am a text generation model."

class MockModel(TextGenerationModel):
    def run_model(self, api_input: Dict[str, List[TextGenerationInput]], headers: Dict[str, str] = None) -> Dict[str, List[TextGenerationOutput]]:
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