from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, \
    T5ForConditionalGeneration
import ktrain
import torch.nn.functional as F
import torch

from summarize import summarize_text

app = Flask(__name__)


def Remove_line(text):
    new = text.replace('\n', ' ')
    return new


# Topic --------------------------------------------------------------------------------
def init_model():
    reloaded_predictor = ktrain.load_predictor('/content/drive/MyDrive/Topics/distilbert_6_topics_1_87.88')
    return reloaded_predictor
    
def predict_topic(text, reloaded_predictor):
  return reloaded_predictor.predict(text)

model = init_model()





# FAKE  --------------------------------------------------------------------------------------------------
fake_tokenizer = AutoTokenizer.from_pretrained("t5-small")
fake_model = AutoModelForSeq2SeqLM.from_pretrained('Final Fake')


def fake_classification(text):
    inputs = fake_tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    pred = fake_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    classification = fake_tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
    return classification


# Sentiment --------------------------------------------------------------------------------------
model_id = "final_sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(model_id)
sent_tuned_model = AutoModelForSequenceClassification.from_pretrained("final_sentiment")


def sentiment_classification(text):
    text = Remove_line(text)
    One_csase = sent_tokenizer([text], return_tensors="pt", padding='max_length', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = sent_tuned_model(**One_csase)

        prediction = F.softmax(outputs.logits, dim=1)

        labels = torch.argmax(prediction, dim=1)
        classification = [sent_tuned_model.config.id2label[label_id] for label_id in labels.tolist()]
    label_mapping = {"LABEL_1": 0, "LABEL_0": -1, "LABEL_2": 1}
    result = ''
    classification_nnn = [label_mapping[label_id] for label_id in classification]
    if (classification_nnn[0] == 0):
        result = "Neutral"
    if (classification_nnn[0] == -1):
        result = "Negative"
    if (classification_nnn[0] == 1):
        result = "Positive"
    return result


# Bias  ----------------------------------------------------------------------------------------------------
bias_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("Combined_trainer_distilbert-base-uncased")


# bias_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def bias_classification(text):
    inputs = bias_tokenizer([text], return_tensors="pt", truncation=True, padding=True
                            )
    with torch.no_grad():
        outputs = fine_tuned_model(**inputs)
    pred = torch.argmax(outputs.logits, dim=-1)
    classification = int(pred)
    if classification == 0:
        return "R-Biased"
    if classification == 1:
        return "L-Biased"
    return "Un-Biased"


#     encoded_input = bias_tokenizer([text], return_tensors='pt', return_token_type_ids=False, truncation=True,
#                                    max_length=512)
#     with torch.no_grad():
#         outputs = bias_tuned_model(**encoded_input)
#         prediction = F.softmax(outputs.logits, dim=1)
#         labels = torch.argmax(prediction, dim=1)
#         classification = [bias_tuned_model.config.id2label[label_id] for label_id in labels.tolist()]
#         if classification[0] == "HYPERPARTISAN":
#             return "Bias"
#         else:
#             return "Un-Bias"


@app.route('/fake', methods=['POST'])
def predict():
    text = request.json['text']
    fake_result = fake_classification(text)
    sent_result = sentiment_classification(text)
    bias = bias_classification(text)
    topic_result = predict_topic(text, model)
    print({'fake': fake_result, 'sentiment': sent_result, "topic": topic_result, "bias": bias})
    return {'fake': fake_result, 'sentiment': sent_result, "topic": topic_result, "bias": bias}


@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.json['text']
    summarize_result = summarize_text(text)
    return summarize_result


if __name__ == '__main__':
    (app.run(host='0.0.0.0', port=5000,debug=True))
