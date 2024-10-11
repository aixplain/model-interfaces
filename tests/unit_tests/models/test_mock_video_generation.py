from unittest.mock import Mock
from aixplain.model_interfaces.schemas.function.function_input import VideoGenerationInput
from aixplain.model_interfaces.schemas.function.function_output import TextSegmentDetails, VideoGenerationOutput 
from aixplain.model_interfaces.interfaces.function_models import VideoGenerationModel
from typing import Dict, List

class TestMockTranslation():
    def test_predict(self):
        data = "Make a video about cute teddy bears doing backflips."
        video_length = 6.0

        video_generation_input = {
            "data": data,
            "video_length": video_length,
            "s3_uri": "mock_s3_uri" # Inserted by backend.
        }

        predict_input = {"instances": [video_generation_input]}
        
        mock_model = MockModel("Mock")
        predict_output = mock_model.predict(predict_input)
        translation_output_dict = predict_output["predictions"][0]

        assert translation_output_dict.data == "mock_s3_uri"
        assert translation_output_dict.details["text"] == "sample details"

class MockModel(VideoGenerationModel):
    def run_model(self, api_input: List[VideoGenerationInput], headers: Dict[str, str] = None) -> List[VideoGenerationOutput]:
        predictions_list = []
        # There's only 1 instance in this case.
        for instance in api_input:
            instance_data = instance.dict()
            model_instance = Mock()
            model_instance.process_data.return_value = "mock_s3_uri"
            result = model_instance.process_data(instance_data["data"])
            model_instance.delete()
            
            # Map back onto TranslationOutput
            data = result
            details = {"text": "sample details"}

            output_dict = {
                "data": data,
                "details": details
            }
            video_generation_output = VideoGenerationOutput(**output_dict)
            predictions_list.append(video_generation_output)
        return predictions_list