import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
import json
import matplotlib as mpl

# pd.set_option('display.max_columns', 500)

def split_dataset(filepath):
    whole_set = pd.read_csv(filepath)
    # Get non-empty text rows, drop null rows, drop duplicate rows
    whole_set = whole_set[whole_set["text"] != ""]
    whole_set.dropna(inplace=True)
    whole_set.drop_duplicates(inplace=True)
    train_set, temp_set = train_test_split(whole_set, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)
    train_set.to_csv('train.csv', index=False)
    val_set.to_csv('val.csv', index=False)
    test_set.to_csv('test.csv', index=False)
    # with open('dataset.csv', 'r', encoding='utf-8') as data,\
    #         open('train.csv', 'a',encoding='utf-8') as train_out,\
    #         open('val.csv', 'a',encoding='utf-8') as val_out, \
    #         open('test.csv', 'a',encoding='utf-8') as test_out:
    #     while line := data.readline():
        
        # reviews = []
        # counter = 0
        # while line := data.readline():
        #     reviews.append(line)
        #     counter += 1
        #     if counter >= 10000:
        #         train, unseen = train_test_split(reviews, test_size=0.2, random_state=42)
        #         val, test = train_test_split(unseen, test_size=0.5, random_state=42)
        #         for x in train:
        #             train_out.write(x)
        #         for x in val:
        #             val_out.write(x)
        #         for x in test:
        #             test_out.write(x)
        #         reviews.clear()
        #         counter = 0


def load_tfidf_vectorizer():
    vectorizer = pickle.load(open("tf_idf_vectorizer.pickle", "rb"))
    return vectorizer

def load_count_vectorizer():
    vectorizer = pickle.load(open("count_vectorizer.pickle", "rb"))
    return vectorizer

def save_tfidf_vectorizer(file_path):
    # Get stopwords and convert to lower.
    
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stop_words_original = set(stopwords.words('english'))
    stop_words = [x.lower() for x in stop_words_original]

    df = pd.read_csv(file_path)
    feature = ["text"]
    labels = ["stars", "useful", "funny", "cool"]
    for label in labels:
        df = df[pd.to_numeric(df[label], errors='coerce').notnull()]
    # Drop negative values
    print(df.dtypes)
    print(df.head)
    # df = df[df.funny >= 0]
    # df = df[df.cool >= 0]
    # df = df[df.useful >= 0]
    # df = df[df.stars >= 1]
    # df = df[df.stars <= 5]

    df["funny"] = df["funny"].astype(int)
    df["cool"] = df["cool"].astype(int)
    df["useful"] = df["useful"].astype(int)
    df["stars"] = df["stars"].astype(int)
    print(df.dtypes)
    df = df[(df["funny"] >= 0) &
            (df["cool"] >= 0) &
            (df["useful"] >= 0) &
            (df["stars"] >= 1 ) & (df["stars"] <= 5)]
    
    
    df["funny"].where(df["funny"] <= 5, 5, inplace=True)
    df["cool"].where(df["cool"] <= 5, 5, inplace=True)
    df["useful"].where(df["useful"] <= 5, 5, inplace=True)
    
    df['text'] = df['text'].apply(text_preprocess)

    df['text'] = df['text'].apply(
            lambda x: ' '.join([word for word in w_tokenizer.tokenize(x) if word not in stop_words]))
    
    X = df[feature].astype('string')
    y = df[labels]

    # modified Vectorizer to look for bigrams, limit to 150 most frequent features
    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1,2))

    print("Fitting Vectorizer...")
    vectorizer.fit(X.text)
    print("Saving Vectorizer")
    pickle.dump(vectorizer, open("tf_idf_vectorizer.pickle", "wb"))

def save_count_vectorizer(file_path):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stop_words_original = set(stopwords.words('english'))
    stop_words = [x.lower() for x in stop_words_original]

    df = pd.read_csv(file_path)
    feature = ["text"]
    labels = ["stars", "useful", "funny", "cool"]
    for label in labels:
        df = df[pd.to_numeric(df[label], errors='coerce').notnull()]
    # Drop negative values
    df["funny"] = df["funny"].astype(int)
    df["cool"] = df["cool"].astype(int)
    df["useful"] = df["useful"].astype(int)
    df["stars"] = df["stars"].astype(int)
    print(df.dtypes)
    df = df[(df["funny"] >= 0) &
            (df["cool"] >= 0) &
            (df["useful"] >= 0) &
            (df["stars"] >= 1 ) & (df["stars"] <= 5)]
    
    df['text'] = df['text'].apply(text_preprocess)

    df['text'] = df['text'].apply(
            lambda x: ' '.join([word for word in w_tokenizer.tokenize(x) if word not in stop_words]))
    X = df[feature].astype('string')
    y = df[labels]

    # modified Vectorizer to look for bigrams, limit to 150 most frequent features
    vectorizer = CountVectorizer(max_features=150, max_df=0.8, min_df=10,ngram_range=(1,1))

    print("Fitting Vectorizer...")
    vectorizer.fit(X.text)
    print("Saving Vectorizer")
    pickle.dump(vectorizer, open("count_vectorizer.pickle", "wb"))

def preprocessing(file_path, tfidf_vectorize=True):
    df = pd.read_csv(file_path)  # 10000 lines x 560 = total
    feature = ["text"]
    labels = ["stars", "useful", "funny", "cool"]
    counter = 0

    print("Loading Vectorizer")
    vectorizer = load_tfidf_vectorizer()
    # vectorizer = load_count_vectorizer()

    for label in labels:
        df = df[pd.to_numeric(df[label], errors='coerce').notnull()]
    df["funny"] = df["funny"].astype(int)
    df["cool"] = df["cool"].astype(int)
    df["useful"] = df["useful"].astype(int)
    df["stars"] = df["stars"].astype(int)
    df["funny"] = df["funny"] + 1
    df["cool"] = df["cool"] + 1
    df["useful"] = df["useful"] + 1
    df["funny"].where(df["funny"] <= 5, 5, inplace=True)
    df["cool"].where(df["cool"] <= 5, 5, inplace=True)
    df["useful"].where(df["useful"] <= 5, 5, inplace=True)
    X = df[feature].astype('string')
    y = df[labels]

    # X['text'] = X['text'].apply(text_preprocess)
    X_vectorize = vectorizer.transform(X.text)
    
    # convert vectorized sparse matrix to dataframe
    X_df = pd.DataFrame(X_vectorize.toarray(), columns=vectorizer.get_feature_names_out())

    df.drop(["text"], axis=1, inplace=True)
    comb = pd.concat([y, X_df], axis=1)
    print(comb)
    print("Saving Vectorized Data...")
    comb.to_csv("hand_anno_vec.csv", index=False) # change this when vectorizing different data sets


def text_preprocess(value):
    try:
        # # Regrex for letters, numbers, and space
        # regex = re.compile('[^a-zA-Z0-9 ]')
        # # Replace with blank space to match stop_words format.
        # word = regex.sub(' ', value)
        # # Lower case
        # word1 = word.lower()

        # Regrex for letters, numbers, and space
        text = re.sub(r'[^A-Za-z0-9 ]+', ' ', value)
        text = re.sub('https?://\S+|www\.\S+', '', text)  # removing URL links
        # Lower case
        text = text.lower()

        return text

        # return word1

    except ValueError:
        return np.NaN

def json_convert_to_csv(infile, outfile):
    try:
        json_data = pd.read_json(infile, lines=True, chunksize=20000)
    except FileNotFoundError:
        print("File not found.")
        exit()
    for chunk in json_data:
        try:
            chunk = chunk[["stars", "useful","funny","cool", "text"]]
            chunk.to_csv(outfile, mode="a", index=False)
        except KeyError as e:
            print(e)

if __name__ == '__main__':
    start_time = time.time()
    # Call this function to convert a JSON dataset to csv
    # json_convert_to_csv("yelp_academic_dataset_review.json", "dataset.csv")

    # Split a CSV dataset into testing, training, and validation sets
    # split_dataset('dataset.csv')

    # Create, train, & save a Tf-Idf vectorizer
    # save_tfidf_vectorizer("train.csv")

    # save_count_vectorizer("test.csv")
    preprocessing('hand_annotated_dataset.csv',tfidf_vectorize=True)

    print("END")
    print(f"--- {(time.time() - start_time):.2f} seconds ---")




