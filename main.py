import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)


@app.route("/answer", methods=["POST"])
def answer_question():
    data = request.get_json()
    question = data["question"]
    context = data["context"]

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    start_scores, end_scores = model(**inputs)

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1

    answer = tokenizer.decode(input_ids[answer_start:answer_end], skip_special_tokens=True)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run()
