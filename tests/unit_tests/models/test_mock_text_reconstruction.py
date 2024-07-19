from unittest.mock import Mock
from aixplain.model_interfaces.schemas.function.function_input import TextReconstructionInput
from aixplain.model_interfaces.schemas.function.function_output import TextReconstructionOutput
from aixplain.model_interfaces.interfaces.function_models import TextReconstructionModel
from typing import Dict, List

class TestMockTextReconstruction():
    def test_predict(self):
        data = "Text to reconstruct."
        supplier = "mockGpt"
        function = "text-reconstruction"
        version = ""
        language = "en"


        input_dict = {
            "data": data,
            "supplier": supplier,
            "function": function,
            "version": version,
            "language": language
        }

        predict_input = {"instances": [input_dict]}
        
        mock_model = MockModel("Mock")
        predict_output = mock_model.predict(predict_input)
        output_dict = predict_output["predictions"][0]

        assert output_dict["data"] == "This is a text reconstruction."

class MockModel(TextReconstructionModel):
    def run_model(self, api_input: Dict[str, List[TextReconstructionInput]], headers: Dict[str, str] = None) -> Dict[str, List[TextReconstructionOutput]]:
        instances = api_input["instances"]
        predictions_list = []
        # There's only 1 instance in this case.
        for instance in instances:
            instance_data = instance.dict()
            model_instance = Mock()
            model_instance.process_data.return_value = "This is a text reconstruction."
            result = model_instance.process_data(instance_data["data"])
            model_instance.delete()
            
            # Map back onto TextReconstructionOutputs
            data = result

            output_dict = {
                "data": data,
            }
            output = TextReconstructionOutput(**output_dict)
            predictions_list.append(output)
        predict_output = {"predictions": predictions_list}
        return predict_output