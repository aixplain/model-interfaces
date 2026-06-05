# عينة تحسين الكلام باستخدام DTLN

تم أخذ النموذج مفتوح المصدر DTLN من مستودع GitHub عبر [الرابط](https://github.com/breizhn/DTLN).

## الإعداد

### تنزيل ملف النموذج

استنسخ مستودع model-interfaces وانتقل إلى المثال المرجعي لتحسين الكلام:

```
git clone git@github.com:aixplain/aixplain-models-internal.git
cd aixplain-models-internal/docs/user/samples/speech-enhancement
```

قم بتنزيل ملف النموذج باستخدام الأوامر التالية:
```
wget https://aixplain-kserve-models-dev.s3.amazonaws.com/serving-models/sample-models/speech-enhancement/dtln/dtln/saved_model.pb
wget https://aixplain-kserve-models-dev.s3.amazonaws.com/serving-models/sample-models/speech-enhancement/dtln/dtln/variables/variables.data-00000-of-00001
wget https://aixplain-kserve-models-dev.s3.amazonaws.com/serving-models/sample-models/speech-enhancement/dtln/dtln/variables/variables.index
```

ضع الملفات في مجلد باسم `dtln` في الدليل الحالي. يجب وضع الملفات وفق هيكل الشجرة التالي:
```
dtln
| - saved_model.pb
| - variables
    | - variables.data-00000-of-00001
    | - variables.index
```

### تثبيت المتطلبات

```
sudo apt-get install ffmpeg

# Install model-interfaces from GitHub, preferably by using a virtualenv
pip install -e 'git+https://$GH_ACCESS_TOKEN@github.com/aixplain/aixplain-models-internal.git@master#egg=model_interfaces'

pip install -r src/requirements.txt
```

- GH_ACCESS_TOKEN: أنشئ رمز وصول GitHub من حسابك يتيح استنساخ مستودع model_interfaces

يمكن العثور على التوثيق الخاص بإنشاء رمز الوصول الشخصي لـ GitHub [هنا](https://docs.github.com/en/enterprise-server@3.4/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

## تشغيل النموذج محليًا

### الاختبار باستخدام ملفات الاختبار والعينات المتوفرة

```
ASSET_DIR=. ASSET_URI=dtln pytest
```

### تقديم النموذج باستخدام خادم ويب

```
ASSET_DIR=. ASSET_URI=dtln python src/model.py
```

### اختبار خادم النموذج

```
python src/sample_request.py
```

## بناء وتشغيل النموذج باستخدام Docker

انتقل إلى المثال المرجعي لتحسين الكلام:

```
cd docs/user/samples/speech-enhancement
```

### بناء حاوية النموذج

```
docker build --build-arg GH_ACCESS_TOKEN=<TOKEN_FROM_GITHUB_ACCOUNT> --build-arg ASSET_URI=<ASSET_URI> . -t 535945872701.dkr.ecr.us-east-1.amazonaws.com/aixmodel-dtln
```

- GH_ACCESS_TOKEN: أنشئ رمز وصول GitHub من حسابك يتيح استنساخ مستودع model_interfaces
- ASSET_URI: اسم النموذج الخاص بك؛ في هذا المثال المرجعي هو 'dtln'.

### تشغيل الحاوية

```
docker run -e ASSET_DIR=/ -e ASSET_URI=dtln -p 8080:8080 535945872701.dkr.ecr.us-east-1.amazonaws.com/aixmodel-dtln
```

<!-- ملاحظات الترجمة: نموذج=Model حسب المسرد | مستودع=Repository حسب المسرد | تثبيت=Install حسب المسرد | تشغيل=Run حسب المسرد | kept EN: DTLN — اسم نموذج مفتوح المصدر | kept EN: GitHub, Docker, pytest — أسماء علامات تجارية/أدوات -->