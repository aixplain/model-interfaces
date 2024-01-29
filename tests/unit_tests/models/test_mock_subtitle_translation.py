from unittest.mock import Mock
from aixplain.model_interfaces.schemas.function.function_input import SubtitleTranslationInput
from aixplain.model_interfaces.schemas.function.function_output import SubtitleTranslationOutput
from aixplain.model_interfaces.interfaces.function_models import SubtitleTranslationModel
from typing import Dict, List

class TestMockSearch():
    def test_predict(self):
        data = "Text to be searched."
        supplier = "mockGpt"
        function = "search"
        version = ""
        source_language = "en"
        dialect_in = "American"
        target_supplier = "mock supplier"
        target_languages = ["fr", "de"]


        input_dict = {
            "data": data,
            "supplier": supplier,
            "function": function,
            "version": version,
            "source_language": source_language,
            "dialect_in": dialect_in,
            "target_supplier": target_supplier,
            "target_languages": target_languages
        }

        search_input = SubtitleTranslationInput(**input_dict)
        predict_input = {"instances": [search_input]}
        
        mock_model = MockModel("Mock")
        predict_output = mock_model.predict(predict_input)
        output_dict = predict_output["predictions"][0]

        assert output_dict["data"] == "This is a subtitle translation."

class MockModel(SubtitleTranslationModel):
    def run_model(self, api_input: Dict[str, List[SubtitleTranslationInput]], headers: Dict[str, str] = None) -> Dict[str, List[SubtitleTranslationOutput]]:
        instances = api_input["instances"]
        predictions_list = []
        # There's only 1 instance in this case.
        for instance in instances:
            instance_data = instance.dict()
            model_instance = Mock()
            model_instance.process_data.return_value = "This is a subtitle translation."
            result = model_instance.process_data(instance_data["data"])
            model_instance.delete()
            
            # Map back onto SubtitleTranslationOutputs
            data = result

            output_dict = {
                "data": data,
            }
            search_output = SubtitleTranslationOutput(**output_dict)
            predictions_list.append(search_output)
        predict_output = {"predictions": predictions_list}
        return predict_output