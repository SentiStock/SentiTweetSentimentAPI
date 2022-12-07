from transformers import AutoTokenizer, AutoConfig
from onnxruntime import InferenceSession
from scipy.special import softmax
import json

import logging
import azure.functions as func

#The HTTP request length is limited to 100 MB (104,857,600 bytes), and the URL length is limited to 4 KB (4,096 bytes). 
# Json in body ( you can uncomment it and then paste to raw tab in Postman)
# {
#     "tweets" : [
#         {
#             "id": "Chinese paper blames Google ",
#             "text": "Chinese paper blames Google over Gmail blocking  http://4-traders.com/GOOGLE-INC-C-16118013/news/GOOGLE-C--Chinese-paper-blames-Google-over-Gmail-blocking-19608660/… $GOOG"
#         },
#         {
#             "id": "Bezos lost money",
#             "text": "Ouch. RT @WSJ: Jeff Bezos lost $7.4 billion in Amazon's worst year since 2008: http://bit.ly/1B6QAFJ $AMZN"
#         },
#         {
#             "id": "Neutral info about stocks",
#             "text": "Starbreakouts newsletter sent. Read why hedge funds are buying $AAPL $CSCO $GOOGL. Check out here http://starbreakouts.com"
#         },
#         {
#             "id": "Amazon y Bezos",
#             "text": "@A_TRON3000 An idiot could run $AMZN at a profit, but only Bezos could run it at a loss. #ThatsACompliment"
#         },
#         {
#             "id": "Neutral data info",
#             "text": "$GOOGL Open Date=Jan-05-2015 Open=527.15 High=527.99 Low=519.32 Close=521.24 Volume=723451 http://investorshangout.com/post/view?id=2524380…"
#         }
#     ]
# }

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def custom_sentiment_function(result_labeled):
    # auto labels ( 'negative', 'neutral', 'positive')
    output = dict(result_labeled)

    #custom_labels
    output['uncertain'] = round(float((output['negative'] + output['positive'])), 4) #always between 0 and 1
    output['compound'] = round((float(((output['positive']+1)**2 - (output['negative']+1)**2)/(output['neutral']+1)))/4, 4) #always between -1 and 1

    return output

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('(Custom log) Python HTTP trigger function started processing a request.')

    req_body_bytes = req.get_body()
    req_body = req_body_bytes.decode("utf-8")

    try:
        tweets = json.loads(req_body)["tweets"]
    except ValueError as e:
        return func.HttpResponse(f"(Custom error) You didn't provide tweets in required format {e}")
    
    LOCAL_PATH = "./models/cardiffnlp-twitter-roberta-base-sentiment-latest"

    #You run the code below only once, just to fetch the required files from huggingface
    #REMOTE_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    #tokenizer = AutoTokenizer.from_pretrained(REMOTE_PATH) # fetching from Huggingface
    #tokenizer.save_pretrained(LOCAL_PATH)
    #config = AutoConfig.from_pretrained(REMOTE_PATH) # fetching from Huggingface
    #config.save_pretrained(LOCAL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
    config = AutoConfig.from_pretrained(LOCAL_PATH)
    session = InferenceSession(f"{LOCAL_PATH}/twitter-roberta-base-sentiment-latest.onnx")

    scored_batch = []

    for tweet_obj in tweets:
        id = tweet_obj['id']
        text = preprocess(tweet_obj['text'])

        # ONNX Runtime expects NumPy arrays as input
        inputs = tokenizer(text, return_tensors="np")
        outputs = session.run(
            output_names=["logits"],
            input_feed=dict(inputs))
        result_array = softmax(outputs[0][0])

        result_labeled = []
        for i in range(result_array.shape[0]):
            label = config.id2label[i]
            score = result_array[i]
            result_labeled.append((label, round(float(score), 4)))  

        result_labeled = custom_sentiment_function(result_labeled)

        scored_batch.append((id, result_labeled))

    return func.HttpResponse(json.dumps(scored_batch), mimetype="application/json")
