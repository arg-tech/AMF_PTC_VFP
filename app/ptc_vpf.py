from datasets import Dataset
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import json
from amf_fast_inference import model
from xaif_eval import xaif


TOKENIZER = AutoTokenizer.from_pretrained("raruidol/ArgumentMining-EN-VFP")
# MODEL = AutoModelForSequenceClassification.from_pretrained("raruidol/ArgumentMining-EN-VFP")
MODEL_ID = "raruidol/ArgumentMining-EN-VFP"
LOADER = model.ModelLoader(MODEL_ID)
PRUNED_MODEL = LOADER.load_model()

def preprocess_data(filexaif):
    proposition_ids = []
    data = {'text': []}

    for node in filexaif['nodes']:
        if node['type'] == 'I':
            ident = node['nodeID']
            data['text'].append(node['text'])
            proposition_ids.append(ident)

    final_data = Dataset.from_dict(data)

    return final_data, proposition_ids


def tokenize_sequence(samples):
    return TOKENIZER(samples["text"], padding=True, truncation=True)


def pipeline_predictions(pipeline, data):
    labels = []
    pipeline_input = []
    for i in range(len(data['text'])):
        sample = data['text'][i]
        pipeline_input.append(sample)

    outputs = pipeline(pipeline_input)
    for out in outputs:
        if out['score'] > 0.8:
            labels.append(out['label'])
        else:
            labels.append('None')

    return labels


def make_predictions(trainer, tknz_data):
    predicted_logprobs = trainer.predict(tknz_data)
    predicted_labels = np.argmax(predicted_logprobs.predictions, axis=-1)

    return predicted_labels


def output_xaif(idents, labels, fileaif):
    #mapping_label = {0: "Value", 1: "Fact", 2: "Policy"}

    if "propositionClassifier" not in fileaif:
        fileaif['propositionClassifier'] = {'nodes': []}
    else:
        if 'nodes' not in fileaif['propositionClassifier']:
            fileaif['propositionClassifier']['nodes'] = []

    for i in range(len(labels)):
        if labels[i] is not None:
            lb = labels[i]
            ident = idents[i]
            fileaif['propositionClassifier']['nodes'].append({"nodeID": ident, "propType": lb})
    return fileaif


def proposition_classification(xaif):

    # Generate a HF Dataset from all the "I" node pairs to make predictions from the xAIF file 
    # and a list of tuples with the corresponding "I" node ids to generate the final xaif file.
    dataset, ids = preprocess_data(xaif['AIF'])

    # Inference Pipeline
    pl = pipeline("text-classification", model=PRUNED_MODEL, tokenizer=TOKENIZER)

    # Predict the list of labels for all the pairs of "I" nodes.
    labels = pipeline_predictions(pl, dataset)

    # Prepare the xAIF output file.
    out_xaif = output_xaif(ids, labels, xaif)

    return out_xaif


# DEBUGGING:
if __name__ == "__main__":
    ff = open('../data.json', 'r')
    content = json.load(ff)
    # print(content)
    out = proposition_classification(content)
    # print(out)
    with open("../data_out.json", "w") as outfile:
        json.dump(out, outfile, indent=4)

