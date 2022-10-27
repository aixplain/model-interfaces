# DTLN Speech Enhancement sample

The open source model DTLN was taken from the github repository in the [link](https://github.com/breizhn/DTLN).

## Setup

### Download the model file

Navigate to the speech-enhancement reference example:

```
cd docs/user/samples/speech-enhancement
```

Download the model file using the commands
```
wget https://aixplain-kserve-models-dev.s3.amazonaws.com/serving-models/sample-models/speech-enhancement/dtln/dtln/saved_model.pb
wget https://aixplain-kserve-models-dev.s3.amazonaws.com/serving-models/sample-models/speech-enhancement/dtln/dtln/variables/variables.data-00000-of-00001
wget https://aixplain-kserve-models-dev.s3.amazonaws.com/serving-models/sample-models/speech-enhancement/dtln/dtln/variables/variables.index
```

Place it in a folder called `dtln` in the current directory. The files should be placed in the following tree's structure:
```
dtln
| - saved_model.pb
| - variables
    | - variables.data-00000-of-00001
    | - variables.index
```

### Install requirements

```
sudo apt-get install ffmpeg

# Install aixplain-models from GitHub
pip install https://github.com/aixplain/aixplain-models-internal.git

pip install -r requirements.txt # Preferably using a virtualenv
```

## Run the model locally

### Test using the given test and samples

```
MODEL_DIR=. MODEL_URI=dtln pytest
```

### Serve the model using a webserver

```
MODEL_DIR=. MODEL_URI=dtln python src/model.py
```

### Test the model server

```
python src/sample_request.py
```

## Build and Run the model with docker

Navigate to the speech-enhancement reference example:

```
cd docs/user/samples/speech-enhancement
```

### Build the model's container

```
docker build --build-arg GH_API_KEY=<KEY_FROM_GITHUB_ACCOUNT> --build-arg MODEL_URI=<MODEL_URI> . -t 535945872701.dkr.ecr.us-east-1.amazonaws.com/aixmodel-dtln
```

- GH_API_KEY: Generate a GitHub API key from your account that can clone aixplain_models repository
- MODEL_URI: Your model's name; 'dtln' in this reference example.

### Run the container

```
docker run -e MODEL_DIR=/ -e MODEL_URI=dtln -p 8080:8080 535945872701.dkr.ecr.us-east-1.amazonaws.com/aixmodel-dtln
```
