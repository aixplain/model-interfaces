from unittest.mock import Mock
from aixplain.model_interfaces.schemas.function.function_input import SearchInput
from aixplain.model_interfaces.schemas.function.function_output import SearchOutput
from aixplain.model_interfaces.interfaces.function_models import SearchModel
from typing import Dict, List

class TestMockSearch():
    def test_predict(self):
        data = "Text to be searched."
        supplier = "mockGpt"
        function = "search"
        version = ""
        language = "en"
        script = ""
        supplier_model_id = "mockID"


        input_dict = {
            "data": data,
            "supplier": supplier,
            "function": function,
            "version": version,
            "language": language,
            "script": script,
            "supplier_model_id": supplier_model_id
        }

        predict_input = {"instances": [input_dict]}
        
        mock_model = MockModel("Mock")
        predict_output = mock_model.predict(predict_input)
        output_dict = predict_output.predictions[0]

        assert output_dict.data == "This is a search output."

class MockModel(SearchModel):
    def run_model(self, api_input: List[SearchInput], headers: Dict[str, str] = None) -> List[SearchOutput]:
        predictions_list = []
        # There's only 1 instance in this case.
        for instance in api_input:
            instance_data = instance.dict()
            model_instance = Mock()
            model_instance.process_data.return_value = "This is a search output."
            result = model_instance.process_data(instance_data["data"])
            model_instance.delete()
            
            # Map back onto SearchOutputs
            data = result

            output_dict = {
                "data": data,
            }
            search_output = SearchOutput(**output_dict)
            predictions_list.append(search_output)
        return predictions_list