from aixplain.model_interfaces.schemas.function.function_input import (
    APIInput,
    AudioEncoding,
    AudioConfig,
    TranslationInput,
    SpeechRecognitionInput,
    DiacritizationInput,
    ClassificationInput,
    SpeechEnhancementInput,
    SpeechSynthesisInput,
    TextToImageGenerationInput,
    TextGenerationInput
)

from aixplain.model_interfaces.schemas.metric.metric_input import(
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

from aixplain.model_interfaces.interfaces.function_models import(
    TranslationModel,
    SpeechRecognitionModel,
    DiacritizationModel,
    ClassificationModel,
    SpeechEnhancementModel,
    SpeechSynthesis,
    TextToImageGeneration,
    TextGenerationModel,
    TextGenerationChatModel
)

from aixplain.model_interfaces.interfaces.metric_models import(
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
    SpeechSynthesis,
    TextToImageGeneration,
    TextGenerationModel,
    TextGenerationChatModel
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
    SpeechSynthesisInput,
    TextToImageGenerationInput,
    TextGenerationInput
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