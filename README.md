# SentiTweetSentimentAPI

source ./.venv/bin/activate
az login --use-device-code
rm -rf .python_packages
pip install  --target="./.python_packages/lib/site-packages"  -r requirements.txt
func azure functionapp publish SentiTweetSentimentAPI --build remote