import pathlib
import typing

import pydantic
import pytest

import aixplain.model_interfaces
from aixplain.model_interfaces.schemas.function.function_input import Segment, SegmentationInput
from aixplain.model_interfaces.schemas.function.function_output import SegmentationOutput
from aixplain.model_interfaces.interfaces.function_models import SegmentationModel
from typing import Dict, List


SEGMENTS = [{"segment_id": 0, "start": 0.0, "end": 1.5, "url": "https://example.com/0.wav"}]


class MockModel(SegmentationModel):
    def run_model(self, api_input: List[SegmentationInput], headers: Dict[str, str] = None) -> List[SegmentationOutput]:
        return [SegmentationOutput(data=instance.data, details=instance.details) for instance in api_input]


class TestMockSegmentation():
    @pytest.mark.parametrize(
        "extra",
        [{}, {"details": None}, {"details": SEGMENTS}],
        ids=["details-omitted", "details-null", "details-list"],
    )
    def test_predict(self, extra):
        instance = {"data": "https://example.com/audio.wav", **extra}
        predict_output = MockModel("Mock").predict({"instances": [instance]})
        output = predict_output["predictions"][0]

        assert output.data == instance["data"]
        if extra.get("details"):
            assert output.details == [Segment(**s) for s in SEGMENTS]
        else:
            assert output.details is None

    def test_output_without_details(self):
        output = SegmentationOutput(data="https://example.com/audio.wav")
        assert output.details is None


def _schema_models():
    package_root = pathlib.Path(aixplain.model_interfaces.__file__).parent
    for path in sorted((package_root / "schemas").rglob("*.py")):
        if path.name == "__init__.py":
            continue
        relative = path.relative_to(package_root.parent.parent).with_suffix("")
        module = __import__(".".join(relative.parts), fromlist=["*"])
        for obj in vars(module).values():
            if (isinstance(obj, type) and issubclass(obj, pydantic.BaseModel)
                    and obj.__module__ == module.__name__):
                yield obj


def test_no_required_optional_fields_in_schemas():
    """Under pydantic v2 an ``Optional[X]`` field with no default is required.
    Every schema field that accepts ``None`` must also have a default."""
    models = list(_schema_models())
    assert models, "no schema models found"
    offenders = [
        f"{model.__module__}.{model.__name__}.{name}"
        for model in models
        for name, field in model.model_fields.items()
        if field.is_required() and type(None) in typing.get_args(field.annotation)
    ]
    assert offenders == []
