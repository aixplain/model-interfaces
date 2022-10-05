from unittest.mock import Mock
from aixplain_models.schemas.function_input import SpeechRecognitionInput
from aixplain_models.schemas.function_output import TextSegmentDetails, SpeechRecognitionOutput 
from aixplain_models.interfaces.function_models import SpeechRecognitionModel
from typing import Dict, List

class TestMockSpeechRecognition():
    def test_predict(self):
        data = "https://testaudio.com"
        supplier = "mockVoice"
        function = "speech-recognition"
        version = ""
        language = "English"

        input_dict = {
            "data": data,
            "supplier": supplier,
            "function": function,
            "version": version,
            "language": language
        }

        speech_recognition_input = SpeechRecognitionInput(**input_dict)
        predict_input = {"instances": [speech_recognition_input]}
        
        mock_model = MockModel("Mock")
        predict_output = mock_model.predict(predict_input)
        output_dict = predict_output["predictions"][0].dict()

        assert output_dict["data"] == "This is a test transcription"
        assert output_dict["details"]["text"] == "This is a test transcription"
        assert output_dict["details"]["confidence"] == 0.7

class MockModel(SpeechRecognitionModel):
    def predict(self, api_input: Dict[str, List[SpeechRecognitionInput]]) -> Dict[str, List[SpeechRecognitionOutput]]:
        instances = api_input["instances"]
        predictions_list = []
        # There's only 1 instance in this case.
        for instance in instances:
            instance_data = instance.dict()
            model_instance = Mock()
            model_instance.process_data.return_value = ("This is a test transcription", 0.7)
            result, confidence = model_instance.process_data(instance_data["data"])
            model_instance.delete()
            
            # Map back onto SpeechRecognitionOutput
            data = result
            details = {"text": result, "confidence": confidence}

            output_dict = {
                "data": data,
                "details": TextSegmentDetails(**details)
            }
            speech_recognition_output = SpeechRecognitionOutput(**output_dict)
            predictions_list.append(speech_recognition_output)
        predict_output = {"predictions": predictions_list}
        return predict_output