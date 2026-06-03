# كيفية تطوير نموذج مستضاف على aiXplain؟

## تطوير النموذج

تنظّم حزمة model-interfaces شيفرة النموذج الخاص بك بتنسيق موحّد بهدف نشر هذه النماذج على نسخ استضافة النماذج في aiXplain. يغطّي الوصف التالي كيفية تنظيم النموذج المستضاف على aiXplain.

### هيكل مجلد النموذج

يجب أن يكون هيكل المجلد على النحو التالي:
```
src
│   model.py
│   bash.sh [Optional]
│   requirements.txt [Optional]
|   Addtional files required by your code [Optional]
```

### مجلد عناصر النموذج

قد يعتمد النموذج المستضاف على ملفات لتحميل المعاملات أو التهيئات أو عناصر النموذج الأخرى. أنشئ مجلدًا للنموذج يحمل نفس الاسم المحدد في `ASSET_URI` وضَع جميع عناصر النموذج التابعة في هذا المجلد.

ملاحظة:
1. سيتم الوصول إلى محتويات هذا المجلد أو تحميلها بواسطة دالة التحميل في فئة النموذج.
2. يكون المتغير البيئي `ASSET_URI` افتراضيًا بالقيمة `asset`.

### تنفيذ ملف model.py

يجب أن يكون كل نموذج مستضاف نسخة من نموذج aiXplain قائم على الدوال. إذا كان النموذج الذي تبنيه هو نموذج ترجمة، على سبيل المثال، فيجب أن يرث تنفيذ فئة النموذج واجهة الفئة TranslationModel كما هو موضح أدناه.

```
from aixplain.model_interfaces.interfaces.function_models import TranslationModel

class ExampleTranslationModel(TranslationModel):
```

يمكن استيراد جميع الواجهات عبر الحزمة `aixplain.model_interfaces`، والتي يمكن تثبيتها كاعتمادية إضافية لحزمة aiXplain SDK الرئيسية باستخدام الأمر `pip install aixplain[model-builder]`.

لتنفيذ واجهة النموذج، عرّف الدوال التالية:

#### دالة التحميل

نفّذ دالة التحميل لتحميل جميع عناصر النموذج من مجلد النموذج المحدد في `ASSET_URI`. يمكن استخدام عناصر النموذج المحمّلة هنا أثناء وقت التنبؤ، أي عند تنفيذ run_model().
عيّن القيمة self.ready إلى 'True' للإشارة إلى أن التحميل قد تم بنجاح.

```
    def load(self):
        model_path = AssetResolver.resolve_path()
        if not os.path.exists(model_path):
            raise ValueError('Model not found')
        self.model = pickle.load(os.path.join(model_path, 'model.pkl'))
        self.ready = True
```

#### دالة تشغيل النموذج

يجب أن تحتوي دالة تشغيل النموذج على المنطق البرمجي للحصول على تنبؤ من النموذج المحمّل.

المدخل:
مدخل دالة تشغيل النموذج هو قاموس. يحتوي هذا القاموس على مفتاح ("instances") يتضمن قيمًا في قائمة تحتوي على فئات فرعية من APIInput مبنية على دوال الذكاء الاصطناعي مثل TranslationInput.

المخرج:
مخرج دالة تشغيل النموذج هو قاموس. يحتوي هذا القاموس على مفتاح ("predictions") يتضمن قيمًا في قائمة تحتوي على فئات فرعية من APIOutput مبنية على دوال الذكاء الاصطناعي مثل TranslationOutput.
يُتوقع أن يُعيد المخرج التنبؤات من النموذج بنفس ترتيب نسخ المدخلات المستلمة.

```
from aixplain.model_interfaces.schemas.function_input import TranslationInput
from aixplain.model_interfaces.schemas.function_output import TranslationOutput

    def run_model(self, api_input: Dict[str, List[TranslationInput]]) -> Dict[str, List[TranslationOutput]]:
        src_text = self.parse_inputs(api_input["instances"])

        translated = self.model.generate(
            **self.tokenizer(
                src_text, return_tensors="pt", padding=True
            )
        )

        predictions = []
        for t in translated:
            data = self.tokenizer.decode(t, skip_special_tokens=True)
            details = TextSegmentDetails(text=data)
            output_dict = {
                "data": data,
                "details": details
            }
            translation_output = TranslationOutput(**output_dict)
            predictions.append(translation_output)
            predict_output = {"predictions": predictions}
        return predict_output
```


### ملفات متطلبات النظام وPython

ملف bash.sh:
يجب أن يتضمن تنفيذ هذا الملف تثبيت أي اعتماديات نظام باستخدام أوامر bash.

ملف requirements.txt:
أدرج جميع حزم Python اللازمة لتشغيل النموذج عن طريق استخراج المتطلبات باستخدام الأمر أدناه

```
pip freeze >> requirements.txt
```
### اختبار النموذج محليًا

شغّل النموذج الخاص بك بالأمر التالي:
```
ASSET_DIR=<path/to/model_artifacts_dir> ASSET_URI=<asset_uri> python -m model
```

أجرِ استدعاء استدلال:

```
ASSET_URI=<asset_uri>
curl -v -H http://localhost:8080/v1/models/$ASSET_URI:predict -d '{"instances": [{"supplier": <supplier>, "function": <function>, "data": <data>}]}'
```

يجب تعديل معامل المدخل في الطلب أعلاه وفقًا لمدخل دالة النموذج المستهدف. راجع [توثيق تعريف مدخل الدالة.](/aixplain/model_interfaces/schemas/function/function_input.py)

### Dockerfile
أنشئ صورة باستخدام نموذج Dockerfile التالي. أضف الميزات حسب الحاجة:
```Dockerfile
FROM python:3.8.10

RUN mkdir /code
WORKDIR /code
COPY . /code/

RUN pip install -r --no-cache-dir requirements.txt

RUN chmod +x /code/bash.sh
RUN ./bash.sh

CMD python -m model
```

### المتغيرات البيئية

 - `ASSET_DIR`: المسار النسبي أو المطلق لمجلد عناصر النموذج (ASSET_URI) على نظامك. القيمة الافتراضية هي المجلد الحالي.
 - `ASSET_URI`: اسم مجلد عناصر النموذج. الاسم الافتراضي هو `asset`.

<!-- ملاحظات الترجمة: استخدام "عناصر النموذج" لترجمة model artifacts | "قاموس" لترجمة dictionary في سياق Python | "فئات فرعية" لترجمة subclass | kept EN: TranslationModel, TranslationInput, TranslationOutput, APIInput, APIOutput — أسماء فئات برمجية | kept EN: ASSET_URI, ASSET_DIR — متغيرات بيئية | kept EN: Dockerfile — اسم ملف تقني معياري -->