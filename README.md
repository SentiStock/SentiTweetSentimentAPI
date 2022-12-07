# SentiTweetSentimentAPI

## How to use
- [SentiTweetSentimentAPI](#sentitweetsentimentapi)
  - [How to use](#how-to-use)
    - [Create Virtual Environment](#create-virtual-environment)
    - [Create Azure Function](#create-azure-function)
    - [Login to Azure](#login-to-azure)
    - [Publish the app](#publish-the-app)
    - [Recommendations](#recommendations)
    - [Helpful resources:](#helpful-resources)

### Create Virtual Environment

    python -m venv .venv
    source .venv/bin/activate

### Create Azure Function

    func init SentiTweetSentimentAPILocalProject --python
    func new --name SentiTweetSentimentAPI --template "HTTP trigger" --authlevel "function"

It will automatically provide some useful bolierplate.

### Login to Azure

    az login --use-device-code

If you don't have az installed run

    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash



source ./.venv/bin/activate

### Publish the app
Either with:
Remote build:

    func azure functionapp publish SentiTweetSentimentAPI --build remote

Local build:

    pip install  --target="./.python_packages/lib/site-packages"  -r requirements.txt
    func azure functionapp publish SentiTweetSentimentAPI --no-build

### Recommendations
We recommend installing the Azure Extension (with Functions) for the VSC.
It is also recommended to use the onnx runtime instead of the standard huggingface pytorch version. To convert this read:

https://huggingface.co/docs/transformers/serialization

The useful commands are either:

Just hidden state as last layer:

    python -m transformers.onnx --model=cardiffnlp/twitter-roberta-base-sentiment-latest onnx/ 

Normal task-specific logits:

    python -m transformers.onnx --model=cardiffnlp/twitter-roberta-base-sentiment-latest --feature=sequence-classification onnx/

We used the following (cause we had already downloaded model):
python3 -m transformers.onnx --model=twitter-roberta-base-sentiment.bin onnx

### Helpful resources:

[ML inference on Azure- dev.to](https://dev.to/azure/why-use-azure-functions-for-ml-inference-ela)
[Serverless deployment of PyTorch on Azure- medium.com](https://medium.com/pytorch/efficient-serverless-deployment-of-pytorch-models-on-azure-dc9c2b6bfee7)
[- Youtube](https://www.youtube.com/watch?v=MCafgeqWMhQ)

Credits to the [used model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
