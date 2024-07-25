import kserve

from typing import Dict, List, Any

from aixplain.model_interfaces.interfaces.project_node import ProjectNode

class TestProjectNode():
    def test_predict(self):
        transcripts = [
            {
                "index": 0,
                "success": True,
                "input_type": "audio",
                "is_url": True,
                "details": {},
                "input_segment_info": [
                    {
                        "length": 11.96,
                        "start": 0.4,
                        "end": 12.36,
                        "speaker": "1",
                        "type": "audio",
                        "node_id": 2,
                        "language": "en",
                        "segment": "https://modelserving-data.s3.amazonaws.com/temp/9e5f5f52-c431-4b7e-99bf-0a3271982676.wav",
                        "is_url": True
                    }
                ],
                "attributes": {
                    "data": "https://aixplain-light-script-node-dev.s3.amazonaws.com/asr_script/audio.mp3?AWSAccessKeyId=AKIAXZSG5AU6Y4QVBO2W&Signature=8sYFcZ%2BfKmwB8r0xcu5qA9rxduE%3D&Expires=1711708650",
                    "input": "https://aixplain-light-script-node-dev.s3.amazonaws.com/asr_script/audio.mp3?AWSAccessKeyId=AKIAXZSG5AU6Y4QVBO2W&Signature=8sYFcZ%2BfKmwB8r0xcu5qA9rxduE%3D&Expires=1711708650"
                }
            }
        ]

        speakers = [
            {
                "index": 0,
                "success": True,
                "input_type": "audio",
                "is_url": True,
                "details": {},
                "input_segment_info": [
                    {
                        "length": 11.96,
                        "start": 0.4,
                        "end": 12.36,
                        "speaker": "1",
                        "type": "audio",
                        "node_id": 2,
                        "language": "en",
                        "segment": "https://modelserving-data.s3.amazonaws.com/temp/9e5f5f52-c431-4b7e-99bf-0a3271982676.wav",
                        "is_url": True
                    }
                ],
                "attributes": {
                    "data": "https://aixplain-light-script-node-dev.s3.amazonaws.com/asr_script/audio.mp3?AWSAccessKeyId=AKIAXZSG5AU6Y4QVBO2W&Signature=8sYFcZ%2BfKmwB8r0xcu5qA9rxduE%3D&Expires=1711708650",
                    "input": "https://aixplain-light-script-node-dev.s3.amazonaws.com/asr_script/audio.mp3?AWSAccessKeyId=AKIAXZSG5AU6Y4QVBO2W&Signature=8sYFcZ%2BfKmwB8r0xcu5qA9rxduE%3D&Expires=1711708650"
                }
            }
        ]

        body = {
            "instances": [
                {
                    "inputs": {
                        "transcripts": transcripts,
                        "speakers": speakers
                    }
                }
            ]
        }
        project_node = MockProjectNode("test-diarization")
        project_node_output = project_node.predict(body)

        assert project_node_output["predictions"][0]["results"][0]["index"] == 0
        assert project_node_output["predictions"][0]["results"][0]["input_segment_info"][0]["length"] == 11.96

class MockProjectNode(ProjectNode):
    def run_script(self, input: Any) -> Any:
        print("Calling run_script")
        inputs = input.inputs
        transcripts = inputs["transcripts"]
        speakers = inputs["speakers"]
        # build the response
        response = []
        print(f'executing script')
        import time
        time.sleep(5)
        for i, transcript in enumerate(transcripts):
            print(f'Merging transcript and diarization at index={i}')
            merge = {"transcript": transcript["attributes"]["data"], "speaker": speakers[i]["input_segment_info"][0]["speaker"]}
            response.append(
                {
                    "index": i,
                    "success": True,
                    "input_type": "text",
                    "is_url": transcript["is_url"],
                    "details": {},
                    "input_segment_info": transcript["input_segment_info"],
                    "attributes": {"data": merge, "input": merge},
                }
            )
        return {"results": response}
if __name__ == "__main__":
    project_node = TestProjectNode("test-diarization")
    # kserve.ModelServer(http_port=8000, grpc_port=8001).start([project_node])
    kserve.ModelServer().start([project_node])