import os
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, \
    TrainingArguments, Trainer
import pickle

metric = evaluate.load("accuracy")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True,
                                                    padding='max_length', max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='utf-8')
    return data


def train(train_data, val_data, name):
    train_texts = train_data['text'].tolist()
    if name == 'stars':
        train_labels = [label - 1 for label in train_data['stars'].tolist()]
        num_labels = 5
    else:
        num_labels = 1
        train_labels = train_data[f'{name}'].tolist()
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_texts = val_data['text'].tolist()
    if name == 'stars':
        val_labels = [label - 1 for label in val_data['stars'].tolist()]
    else:
        val_labels = val_data[f'{name}'].tolist()
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = pd.DataFrame(
        {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'],
         'labels': train_labels})
    val_dataset = pd.DataFrame(
        {'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask'],
         'labels': val_labels})

    train_dataset = Dataset.from_pandas(train_dataset)
    val_dataset = Dataset.from_pandas(val_dataset)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )

    training_args = TrainingArguments(output_dir=f"trainers/{name}_trainer", evaluation_strategy="epoch",
                                      num_train_epochs=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(f'model/{name}_model')
    return model


def test_model(model, test_data, name):
    train_texts = test_data['text'].tolist()
    if name == 'stars':
        test_labels = [label - 1 for label in test_data['stars'].tolist()]
    else:
        test_labels = test_data[f'{name}'].tolist()
    test_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_dataset = pd.DataFrame(
        {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'],
         'labels': test_labels})
    test_dataset = Dataset.from_pandas(test_dataset)
    testing_args = TrainingArguments(output_dir=f'test_results/{name}_results',
                                     num_train_epochs=3)
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        eval_dataset=test_dataset,
        args=testing_args
    )
    model_pred = trainer.predict(test_dataset)
    print(f"Testing Results for {name} Model:\n{model_pred.metrics}")


def run_tests(test_data):

    stars_model = load_model('stars_model')
    test_model(stars_model, test_data, 'stars')

    cool_model = load_model('cool_model')
    test_model(cool_model, test_data, 'cool')

    funny_model = load_model('funny_model')
    test_model(funny_model, test_data, 'funny')

    useful_model = load_model('useful_model')
    test_model(useful_model, test_data, 'useful')


def load_model(model):
    # config_file = os.path.join(os.getcwd(), 'model', model, 'config.json')
    # config = DistilBertConfig.from_json_file(config_file)
    model = DistilBertForSequenceClassification.from_pretrained(os.path.join(os.getcwd(), '../model', model))
    return model


def main():
    stars_data = load_data('pickles/train_star.pickle')
    val_stars_data = load_data('pickle/val.pickle').sample(n=2000, random_state=42)
    train(stars_data, val_stars_data, 'stars')

    cool_data = load_data('pickles/train_cool.pickle')
    val_cool_data = load_data('pickle/val.pickle').sample(n=2000, random_state=42)
    cool_data['cool'] = cool_data['cool'].astype(float)
    train(cool_data, val_cool_data, 'cool')

    funny_data = load_data('pickles/train_funny.pickle')
    val_funny_data = load_data('pickle/val.pickle').sample(n=2000, random_state=42)
    funny_data['funny'] = funny_data['funny'].astype(float)
    train(funny_data, val_funny_data, 'funny')

    useful_data = load_data('pickles/train_useful.pickle')
    val_useful_data = load_data('pickle/val.pickle').sample(n=2000, random_state=42)
    useful_data['funny'] = useful_data['useful'].astype(float)
    train(useful_data, val_useful_data, 'useful')

    
    #RUN THE TESTS
    test_data = load_data('pickles/test.pickle').sample(n=2000, random_state=42)
    run_tests(test_data)


if __name__ == '__main__':
    main()
