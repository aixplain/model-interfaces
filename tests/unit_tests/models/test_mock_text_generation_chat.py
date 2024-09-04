from unittest.mock import Mock
from aixplain.model_interfaces.schemas.function.function_input import TextGenerationInput
from aixplain.model_interfaces.schemas.function.function_output import TextGenerationOutput
from aixplain.model_interfaces.interfaces.function_models import TextGenerationChatModel, TextGenerationChatTemplatizeInput
from aixplain.model_interfaces.schemas.modality.modality_input import TextListInput
from typing import Dict, List, Text

class TestMockTextGeneration():
    def test_predict(self):
        predict_input = {
            "instances": [
                {
                    "data": "How many cups in a liter?",
                    "max_new_tokens": 200,
                    "top_p": 0.92,
                    "top_k": 1,
                    "num_return_sequences": 1
                }
            ],
            "function": "predict"
        }
        
        mock_model = MockModel("Mock")
        predict_output = mock_model.predict(predict_input)
        predictions = predict_output["predictions"][0]

        assert predictions.data == "I am a text generation model."

    def test_tokenize(self):
        tokenize_input = {
            # provide a list of test instances
            "instances": [
                {
                    "data": ["Hello world", "Hello world again"]
                }
            ],
            "function": "TOKENIZE"
        }
        mock_model = MockModel("Mock")
        token_counts_list = mock_model.predict(tokenize_input)
        print(f"Token counts: {token_counts_list}")

        assert token_counts_list["token_counts"][0] == [11, 17]

    def test_templatize(self):
        data_to_be_templatized = [
            {
                "role": "user",
                "content": "Hello, how are you?"
            },
            {
                "role": "assistant",
                "content": "I'm doing great. How can I help you today?"
            },
            {
                "role": "user",
                "content": "I'd like to show off how chat templating works!"
            },
            {
                "role": "system",
                "content": "I'd like to show off how chat templating works!"
            }
        ]
        templatize_input = {
            "instances": [
                {
                    "data": data_to_be_templatized
                }
            ],
            "function": "TEMPLATIZE"
        }

        mock_model = MockModel("Mock")
        templatized_text = mock_model.predict(templatize_input)
        
        assert templatized_text["prompts"][0] == f"Mock template: {str(data_to_be_templatized)}"
        # for i in range(len(data_to_be_templatized)):
        #     print(f"templatized_text: {templatized_text}")
        #     assert templatized_text["prompts"][i] == f"Mock template: {str(data_to_be_templatized[i])}"


class MockModel(TextGenerationChatModel):
    def run_model(self, api_input: List[TextGenerationInput], headers: Dict[str, str] = None) -> List[TextGenerationOutput]:
        print(f"API INPUT: {api_input}")
        instances = api_input
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
        predict_output = predictions_list
        return predict_output
    
    def tokenize(self, api_input: List[TextListInput], headers: Dict[str, str] = None) -> List[List[int]]:
        token_counts_list = []
        for instance in api_input:
            token_counts = [len(message) for message in instance.data]
            token_counts_list.append(token_counts)
        return token_counts_list

    def templatize(self, api_input: List[TextGenerationChatTemplatizeInput], headers: Dict[str, str] = None) -> List[Text]:
        template_text_list = []
        for instance in api_input:
            templatized_text = f"Mock template: {str(instance.data)}"
            template_text_list.append(templatized_text)
        return template_text_list