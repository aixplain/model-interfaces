from aixplain_models.schemas.function_input import (
    APIInput,
    AudioEncoding,
    AudioConfig,
    TranslationInput,
    SpeechRecognitionInput,
    DiacritizationInput,
    ClassificationInput,
    SpeechEnhancementInput,
    SpeechSynthesisInput
)

from aixplain_models.schemas.function_output import(
    APIOutput,
    WordDetails,
    TextSegmentDetails,
    Label,
    TranslationOutput,
    SpeechRecognitionOutput,
    DiacritizationOutput,
    ClassificationOutput,
    SpeechEnhancementOutput
)

from aixplain_models.schemas.metric_input import(
    MetricInput,
    MetricAggregate,
    TextGenerationSettings,
    AudioGenerationSettings,
    TextGenerationMetricInput,
    ReferencelessTextGenerationMetricInput,
    AudioGenerationMetricInput,
    ReferencelessAudioGenerationMetricInput,
    ClassificationMetricInput,
    NamedEntityRecognitionElement,
    NamedEntityRecognitionMetricInput
)

from aixplain_models.schemas.metric_output import(
    MetricOutput,
    TextGenerationMetricOutput,
    ReferencelessTextGenerationMetricOutput,
    AudioGenerationMetricOutput,
    ReferencelessAudioGenerationMetricOutput,
    ClassificationMetricOutput,
    NamedEntityRecognitionMetricOutput
)

from aixplain_models.interfaces.function_models import(
    TranslationModel,
    SpeechRecognitionModel,
    DiacritizationModel,
    ClassificationModel,
    SpeechEnhancementModel,
    SpeechSynthesis
)

from aixplain_models.interfaces.metric_models import(
    TextGenerationMetric,
    ReferencelessTextGenerationMetric,
    ClassificationMetric,
    AudioGenerationMetric,
    ReferencelessAudioGenerationMetric,
    NamedEntityRecognitionMetric
)

function_classes = [
    TranslationModel,
    SpeechRecognitionModel,
    DiacritizationModel,
    ClassificationModel,
    SpeechEnhancementModel,
    SpeechSynthesis
]

function_classes_input = [
    APIInput,
    AudioEncoding,
    AudioConfig,
    TranslationInput,
    SpeechRecognitionInput,
    DiacritizationInput,
    ClassificationInput,
    SpeechEnhancementInput,
    SpeechSynthesisInput
]

metric_classes_input = [
    MetricInput,
    MetricAggregate,
    TextGenerationSettings,
    AudioGenerationSettings,
    TextGenerationMetricInput,
    ReferencelessTextGenerationMetricInput,
    AudioGenerationMetricInput,
    ReferencelessAudioGenerationMetricInput,
    ClassificationMetricInput,
    NamedEntityRecognitionElement,
    NamedEntityRecognitionMetricInput
]

metric_classes = [
    TextGenerationMetric,
    ReferencelessTextGenerationMetric,
    ClassificationMetric,
    AudioGenerationMetric,
    ReferencelessAudioGenerationMetric,
    NamedEntityRecognitionMetric
]

function_input_interface_map = {clazz.__name__.replace("Input", ""): clazz for clazz in function_classes_input}
metric_input_interface_map = {clazz.__name__.replace("Input", ""): clazz for clazz in metric_classes_input}
function_interface_map = {clazz.__name__: clazz for clazz in function_classes}
metric_interface_map = {clazz.__name__: clazz for clazz in metric_classes}