from unittest.mock import Mock
from aixplain.model_interfaces.schemas.function.function_input import FillTextMaskInput
from aixplain.model_interfaces.schemas.function.function_output import FillTextMaskOutput
from aixplain.model_interfaces.interfaces.function_models import FillTextMaskModel
from typing import Dict, List

class TestMockFillTextMask():
    def test_predict(self):
        data = "Text to reconstruct."
        supplier = "mockGpt"
        function = "fill-text-mask"
        version = ""
        language = "en"
        dialect = "American"
        script = "mock script"


        input_dict = {
            "data": data,
            "supplier": supplier,
            "function": function,
            "version": version,
            "language": language,
            "dialect": dialect,
            "script": script
        }

        predict_input = {"instances": [input_dict]}
        
        mock_model = MockModel("Mock")
        predict_output = mock_model.predict(predict_input)
        output_dict = predict_output["predictions"][0]

        assert output_dict.data == "We are filling a text mask."

class MockModel(FillTextMaskModel):
    def run_model(self, api_input: List[FillTextMaskInput], headers: Dict[str, str] = None) -> List[FillTextMaskOutput]:
        predictions_list = []
        # There's only 1 instance in this case.
        for instance in api_input:
            instance_data = instance.dict()
            model_instance = Mock()
            model_instance.process_data.return_value = "We are filling a text mask."
            result = model_instance.process_data(instance_data["data"])
            model_instance.delete()
            
            # Map back onto FillTextMaskOutputs
            data = result

            output_dict = {
                "data": data,
            }
            output = FillTextMaskOutput(**output_dict)
            predictions_list.append(output)
        return predictions_list