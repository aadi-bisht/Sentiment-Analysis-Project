import pandas as pd
import pickle
import numpy as np
from nltk import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import time
import re

from sklearn.feature_extraction.text import CountVectorizer



w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
stop_words_original = set(stopwords.words('english'))
stop_words = [x.lower() for x in stop_words_original]
# english_words = set(nltk.corpus.words.words())
# english_words = [x.lower() for x in english_words_original]


def partial_fit(self , data):
    if(hasattr(self , 'vocabulary_')):
        vocab = self.vocabulary_
    else:
        vocab = {}
    self.fit(data)
    vocab = list(set(vocab.keys()).union(set(self.vocabulary_ )))
    self.vocabulary_ = {vocab[i] : i for i in range(len(vocab))}


def naive_bayes(df, cv, nb_clf, label="stars"):
    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # cv = CountVectorizer(min_df=2, ngram_range=(2, 2), tokenizer=token.tokenize)
    # cv = CountVectorizer(min_df=10, ngram_range=(1, 1), tokenizer=token.tokenize)


    # train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    # train_X = train_df['text']
    # test_X = test_df['text']

    cv.partial_fit(df['text'])
    # X = cv.transform(df['text'])
    # test_X = cv.transform(test_X)

    nb_clf.fit(df, cv, query_label=label)

    return cv, nb_clf


def naive_bayes_test(df, cv, nb_clf, label="stars"):
    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # cv = CountVectorizer(min_df=2, ngram_range=(2, 2), tokenizer=token.tokenize)
    # cv = CountVectorizer(min_df=10, ngram_range=(1, 1), tokenizer=token.tokenize)


    test_X = cv.transform(df['text'])
    test_y_stars = df[label]
    pred = nb_clf.predict(test_X)

    print(f"\nClassifier: Naive Bayes  ---- Label: {label.upper()}")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(test_y_stars, pred) * 100))
    print(metrics.classification_report(test_y_stars, pred))  # labels=attack_columns


def naive_bayes_val(df, cv, nb_clf, label="stars"):
    test_X = cv.transform(df['text'])
    test_y_stars = df[label]
    pred = nb_clf.predict(test_X)

    return metrics.accuracy_score(test_y_stars, pred)



def naive_bayes_train_test(df, cv, nb_clf, label="stars"):
    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # cv = CountVectorizer(min_df=2, ngram_range=(2, 2), tokenizer=token.tokenize)
    # cv = CountVectorizer(min_df=10, ngram_range=(1, 1), tokenizer=token.tokenize)


    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    train_X = train_df['text']
    test_X = test_df['text']
    test_y_stars = test_df[label]

    cv.partial_fit(train_X)
    # X = cv.transform(train_X['text'])
    test_X = cv.transform(test_X)

    nb_clf.fit(train_df, cv, query_label=label)

    pred = nb_clf.predict(test_X)

    print(f"\nClassifier: Naive Bayes  ---- Label: {label.upper()}")
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(test_y_stars, pred) * 100))
    print(metrics.classification_report(test_y_stars, pred))  # labels=attack_columns

    return cv, nb_clf


def preprocessing_naive_bayes(file_path='train.json', chunksize=10000, drop=None):
    json_data = pd.read_json(file_path, lines=True, chunksize=chunksize)  # 10000 lines x 560 = total
    feature = ['text']
    labels = ['stars', 'useful', 'funny', 'cool']
    counter = 0

    for df in json_data:
        # Drop negative values, scale to 0-4 (1-5) for funny, cool, useful
        df = df[df.funny >= 0]
        df = df[df.cool >= 0]
        df = df[df.useful >= 0]
        df = df[df.stars >= 1]
        df = df[df.stars <= 5]

        # Drop reviews with star of certain rating
        if drop == "stars1":
            df = df[df.stars != 1]
        elif drop == "stars2":
            df = df[df.stars != 2]
        elif drop == "stars3":
            df = df[df.stars != 3]
        elif drop == "stars4":
            df = df[df.stars != 4]
        elif drop == "stars5":
            df = df[df.stars != 5]

        df["funny"] = df["funny"] + 1
        df["cool"] = df["cool"] + 1
        df["useful"] = df["useful"] + 1
        df["funny"].where(df["funny"] <= 5, 5, inplace=True)
        df["cool"].where(df["cool"] <= 5, 5, inplace=True)
        df["useful"].where(df["useful"] <= 5, 5, inplace=True)
        df['text'] = df['text'].astype('string')

        df['text'] = df['text'].apply(text_cleaning)

        # print(y.value_counts())
        # print(X.describe().T)
        # df.text.str.len().hist()
        # print(X.head())

        df['text'] = df['text'].apply(
            lambda x: ' '.join([word for word in w_tokenizer.tokenize(x) if word not in stop_words]))
        # X['text'] = X['text'].apply(lambda x: ' '.join([word for word in w_tokenizer.tokenize(x) if word in english_words]))

        # # Working with the most Frequent Words:
        # from collections import Counter
        # cnt = Counter()
        # for text in dt["no_sw"].values:
        #     for word in text.split():
        #         cnt[word] += 1
        # cnt.most_common(10)
        # temp = pd.DataFrame(cnt.most_common(10))
        # temp.columns = ['word', 'count']
        # temp

        wordnet_lem = nltk.stem.WordNetLemmatizer()
        df['text'] = df['text'].apply(wordnet_lem.lemmatize)
        # print("\nAfter stopwords and lemmatize")
        # print(X.head())

        counter += 1
        print(f"Chunk: {counter} / 560")

        X = df[feature]
        y = df[labels]
        yield df, X, y, counter


def text_cleaning(value):
    # Regrex for letters, numbers, and space
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', value)
    text = re.sub('https?://\S+|www\.\S+', '', text)  # removing URL links
    # Lower case
    text = text.lower()

    return text


class NaiveBayesAttempt:
    def __init__(self):
        self.vc_vocab = CountVectorizer()

        self.stars_prior_prob = {"star1": 0, "star2": 0, "star3": 0, "star4": 0, "star5": 0}
        self.star1_prior_count = 0
        self.star2_prior_count = 0
        self.star3_prior_count = 0
        self.star4_prior_count = 0
        self.star5_prior_count = 0

        self.star1_words = {}
        self.star2_words = {}
        self.star3_words = {}
        self.star4_words = {}
        self.star5_words = {}
        self.star1_total_words = 0
        self.star2_total_words = 0
        self.star3_total_words = 0
        self.star4_total_words = 0
        self.star5_total_words = 0
        self.star1_words_prob = {}
        self.star2_words_prob = {}
        self.star3_words_prob = {}
        self.star4_words_prob = {}
        self.star5_words_prob = {}

        # self.features = list
        # self.likelihoods = {}
        # self.class_priors = {}
        # self.pred_priors = {}
        #
        # self.X_train = np.array
        # self.y_train = np.array
        # self.train_size = int
        # self.num_feats = int

    def fit(self, df, cv, query_label='stars'):
        print(f"Vocabulary Size: {len(cv.vocabulary_)}")
        self.vc_vocab = cv.vocabulary_

        star1_X = df.query(f'{query_label}==1')[['text']]
        star1_X_trans = cv.transform(star1_X['text'])
        self.star1_words, self.star1_total_words = self.update_word_count(star1_X_trans, cv, self.star1_words,
                                                                          self.star1_total_words)

        star2_X = df.query(f'{query_label}==2')[['text']]
        star2_X_trans = cv.transform(star2_X['text'])
        self.star2_words, self.star2_total_words = self.update_word_count(star2_X_trans, cv, self.star2_words,
                                                                          self.star2_total_words)

        star3_X = df.query(f'{query_label}==3')[['text']]
        star3_X_trans = cv.transform(star3_X['text'])
        self.star3_words, self.star3_total_words = self.update_word_count(star3_X_trans, cv, self.star3_words,
                                                                          self.star3_total_words)

        star4_X = df.query(f'{query_label}==4')[['text']]
        star4_X_trans = cv.transform(star4_X['text'])
        self.star4_words, self.star4_total_words = self.update_word_count(star4_X_trans, cv, self.star4_words,
                                                                          self.star4_total_words)

        star5_X = df.query(f'{query_label}==5')[['text']]
        star5_X_trans = cv.transform(star5_X['text'])
        self.star5_words, self.star5_total_words = self.update_word_count(star5_X_trans, cv, self.star5_words,
                                                                          self.star5_total_words)

        star1_len = len(star1_X)
        star2_len = len(star2_X)
        star3_len = len(star3_X)
        star4_len = len(star4_X)
        star5_len = len(star5_X)

        self.star1_prior_count += len(star1_X)
        self.star2_prior_count += len(star2_X)
        self.star3_prior_count += len(star3_X)
        self.star4_prior_count += len(star4_X)
        self.star5_prior_count += len(star5_X)

        stars_total_len = star1_len + star2_len + star3_len + star4_len + star5_len
        # print(star1_len)
        # print(star2_len)
        # print(star3_len)
        # print(star4_len)
        # print(star5_len)
        # print(stars_total_len)

        self.stars_prior_prob = {"star1": star1_len/stars_total_len,
                                 "star2": star2_len/stars_total_len,
                                 "star3": star3_len/stars_total_len,
                                 "star4": star4_len/stars_total_len,
                                 "star5": star5_len/stars_total_len}

        # print(self.stars_prior_prob)

        # Calculate probability
        self.star1_words_prob = self.update_probability(self.star1_words, self.star1_total_words, "1")
        self.star2_words_prob = self.update_probability(self.star2_words, self.star2_total_words, "2")
        self.star3_words_prob = self.update_probability(self.star3_words, self.star3_total_words, "3")
        self.star4_words_prob = self.update_probability(self.star4_words, self.star4_total_words, "4")
        self.star5_words_prob = self.update_probability(self.star5_words, self.star5_total_words, "5")

        # y_stars = y['stars']
        # train_X, test_X, train_y_stars, test_y_stars = train_test_split(X, y_stars, stratify=y_stars,
        #                                                                 random_state=42)

        # vocab = cv.get_feature_names_out()
        # vocab2 = cv.vocabulary_
        # print(X)
        # print(cv.get_feature_names_out())
        # print(cv.vocabulary_)
        # X_sum = np.asarray(X.sum(axis=0)).squeeze()
        # print(X_sum)  # gets count of words frequencies (in the same order as vocabulary_ index)

        # print(df['text'].iloc[0])
        # print(df['stars'].iloc[0])
        # print(y.iloc[0])
        # print(y_stars.iloc[0])

        # print(type(X))


    def update_word_count(self, X, cv, label_words, label_total_words):
        # print(len(cv.vocabulary_))
        word_count = dict.fromkeys(cv.vocabulary_, 0)

        # print(star1_X)
        # print(star1_X_trans)
        # print(star1_word_count)
        X_sum = np.asarray(X.sum(axis=0)).squeeze()
        # print(X_sum)
        counter = 0
        for key, value in word_count.items():
            word_count[key] += X_sum[counter]
            counter += 1
        # print(word_count)

        for key, value in word_count.items():
            if key not in label_words:
                label_words[key] = value
            else:
                label_words[key] += value

        label_total_words += sum(X_sum)

        # print(label_words)
        # print(label_total_words)
        return label_words, label_total_words

    def update_probability(self, words, total_words, category_name = "Category"):
        words_prob = dict.fromkeys(words, 0)
        # print(words)
        # print(words_prob)

        star1_total = total_words
        star1_total_len = len(words_prob)
        for key, value in words_prob.items():
            count = words[key]
            words_prob[key] = (1 + count)/(star1_total_len + star1_total)

        # print(words_prob)

        # print("Sum probability for category {} should sum to 1. The actual summation is equal to {}.".format(
        #     category_name, sum(words_prob.values())))

        return words_prob


    def stars_predict(self, sentence):
        word_count_vector = sentence.toarray().squeeze()
        word_count_vector_active = np.where(word_count_vector != 0)[0]
        # print(word_count_vector)
        # print(word_count_vector_active)

        star1_word_prob = []
        star2_word_prob = []
        star3_word_prob = []
        star4_word_prob = []
        star5_word_prob = []

        for index in word_count_vector_active:
            key = get_nth_key(self.vc_vocab, index)
            # print(key)
            # print(self.star1_words_prob[key])
            star1_word_prob.append(self.star1_words_prob[key])
            star2_word_prob.append(self.star2_words_prob[key])
            star3_word_prob.append(self.star3_words_prob[key])
            star4_word_prob.append(self.star4_words_prob[key])
            star5_word_prob.append(self.star5_words_prob[key])

        if len(word_count_vector_active) == 0:
            return 5  # 5 star is highest probability, so return as default guess.

        star1_final_prob = self.stars_prior_prob["star1"] * np.prod(star1_word_prob)
        star2_final_prob = self.stars_prior_prob["star2"] * np.prod(star2_word_prob)
        star3_final_prob = self.stars_prior_prob["star3"] * np.prod(star3_word_prob)
        star4_final_prob = self.stars_prior_prob["star4"] * np.prod(star4_word_prob)
        star5_final_prob = self.stars_prior_prob["star5"] * np.prod(star5_word_prob)

        if star5_final_prob >= star1_final_prob and \
                star5_final_prob >= star2_final_prob and \
                star5_final_prob >= star3_final_prob and \
                star5_final_prob >= star4_final_prob:
            return 5
        elif star4_final_prob >= star1_final_prob and \
                star4_final_prob >= star2_final_prob and \
                star4_final_prob >= star3_final_prob and \
                star4_final_prob >= star5_final_prob:
            return 4
        elif star3_final_prob >= star1_final_prob and \
                star3_final_prob >= star2_final_prob and \
                star3_final_prob >= star4_final_prob and \
                star3_final_prob >= star5_final_prob:
            return 3
        elif star2_final_prob >= star1_final_prob and \
                star2_final_prob >= star3_final_prob and \
                star2_final_prob >= star4_final_prob and \
                star2_final_prob >= star5_final_prob:
            return 2
        elif star1_final_prob >= star2_final_prob and \
                star1_final_prob >= star3_final_prob and \
                star1_final_prob >= star4_final_prob and \
                star1_final_prob >= star5_final_prob:
            return 1
        else:
            print("ERROR? No return?")
            print(star1_final_prob)
            print(star2_final_prob)
            print(star3_final_prob)
            print(star4_final_prob)
            print(star5_final_prob)
            return 5

    def predict(self, X):
        result = [self.stars_predict(x) for x in X]
        return result


def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")


def load_pickle(path):
    with open(path, 'rb') as f:
        y = pickle.load(f)
    return y


def train(label="stars", drop=None, pickle_path='pickles/NaiveBayes_Stars_baseline.pickle',
          cv_path='pickles/Count_Vectorizer_baseline.pickle'):
    # split_dataset()
    # for X_text, y_label in preprocessing.preprocessing('train.json', tfidf_vectorize=False):

    # allow partial fitting for count vectorizor.
    CountVectorizer.partial_fit = partial_fit
    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # cv = CountVectorizer(min_df=100, ngram_range=(1, 1),
    #                      tokenizer=token.tokenize)  # min_df=10, ngram_range=(2, 2) chunksize=10000
    # cv = CountVectorizer(min_df=0.05, max_df=0.4, ngram_range=(1, 2))  # min_df=20, ngram_range=(2, 2) chunksize=10000
    cv = CountVectorizer(stop_words='english', ngram_range=(1, 1))  # baseline default
    # cv = load_pickle('Count_Vectorizer.pickle')
    nb_clf = NaiveBayesAttempt()

    for df, X_text, y_label, counter in preprocessing_naive_bayes(file_path='train.json', chunksize=10000, drop=drop):
        # Process each chunk here
        naive_bayes(df, cv, nb_clf, label=label)
        # naive_bayes_train_test(df, cv, nb_clf, label="stars")
        if counter >= 10:  # Comment this to go through rest of dataset
            break

    with open(cv_path, 'wb') as f:
        pickle.dump(cv, f)

    with open(pickle_path, 'wb') as f:
        pickle.dump(nb_clf, f)


def test(nb_clf, label="stars", cv=None, drop=None):
    # split_dataset()
    # for X_text, y_label in preprocessing.preprocessing('train.json', tfidf_vectorize=False):

    # allow partial fitting for count vectorizor.
    CountVectorizer.partial_fit = partial_fit
    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # cv = load_pickle('Count_Vectorizer.pickle')
    if cv is None:
        cv = load_pickle('pickles/Count_Vectorizer.pickle')

    for df, X_text, y_label, counter in preprocessing_naive_bayes(file_path='test.json', chunksize=10000, drop=drop):
        # Process each chunk here
        naive_bayes_test(df, cv, nb_clf, label=label)
        if counter >= 1:  # Comment this to go through rest of dataset
            break


def train_with_test(label="stars"):
    # split_dataset()
    # for X_text, y_label in preprocessing.preprocessing('train.json', tfidf_vectorize=False):

    # allow partial fitting for count vectorizor.
    CountVectorizer.partial_fit = partial_fit
    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(min_df=10, ngram_range=(2, 2))  # min_df=10, ngram_range=(2, 2) chunksize=10000
    # cv = load_pickle('Count_Vectorizer.pickle')
    nb_clf = NaiveBayesAttempt()

    train_acc_history = []
    val_acc_history = []

    train_df = None

    for val_df, val_X_text, val_y_label, val_counter in preprocessing_naive_bayes(file_path='val.json', chunksize=3000):

        for df, X_text, y_label, counter in preprocessing_naive_bayes(file_path='train.json', chunksize=10000):
            # Process each chunk here
            naive_bayes(df, cv, nb_clf, label=label)
            if train_df is None:
                _, train_val = train_test_split(df, test_size=0.3, random_state=42)
                train_df = train_val
            train_acc = naive_bayes_val(train_val, cv, nb_clf, label=label)
            train_acc_history.append(train_acc)

            val_acc = naive_bayes_val(val_df, cv, nb_clf, label=label)
            val_acc_history.append(val_acc)

            if counter >= 60:  # Comment this to go through rest of dataset
                break

        break

    plot_train_val_accuracy_to_chunks(train_acc_history, val_acc_history, title=f'Naive Bayes Accuracy (ngram = (2, 2), min_df = 10)')

    # plot_train_val_accuracy_to_chunks(train_acc_history, val_acc_history)

    # with open('Count_Vectorizer.pickle', 'wb') as f:
    #     pickle.dump(cv, f)

    # with open(f'NaiveBayes_Stars2.pickle', 'wb') as f:
    #     pickle.dump(nb_clf, f)


def plot_val_accuracy_to_chunks(val_acc, title=f'Naive Bayes Accuracy'):
    plt.plot(val_acc)
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('chunks')
    plt.legend(['Validation'], loc='upper left')
    plt.show()


def plot_train_val_accuracy_to_chunks(train_acc, val_acc, title='Model Accuracy'):
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('chunks')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()



def train_remove_k_highest_frequency(label="stars", k=30, pickle_path="pickles/NaiveBayes_Funny_top_15_dropped2.pickle",
                                     cv_path='pickles/Count_Vectorizer_top_15_dropped.pickle'):
    # split_dataset()
    # for X_text, y_label in preprocessing.preprocessing('train.json', tfidf_vectorize=False):

    # allow partial fitting for count vectorizor.
    CountVectorizer.partial_fit = partial_fit
    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # cv = CountVectorizer(min_df=100, ngram_range=(1, 1), tokenizer=token.tokenize)  # min_df=10, ngram_range=(2, 2) chunksize=10000
    cv = CountVectorizer(min_df=100, ngram_range=(1, 1))  # min_df=10, ngram_range=(2, 2) chunksize=10000
    # cv = load_pickle('Count_Vectorizer_top_15_dropped.pickle')
    nb_clf = NaiveBayesAttempt()

    high_freq_df = get_k_highest_frequency_words(k=k)
    high_freq_words = high_freq_df["WORD"].to_numpy().squeeze()
    print(high_freq_words)

    for df, X_text, y_label, counter in preprocessing_naive_bayes(file_path='train.json', chunksize=10000):

        # Process each chunk here
        # print(X_text)
        # print(y_label)
        df['text'] = df['text'].apply(
            lambda x: ' '.join([word for word in w_tokenizer.tokenize(x) if word not in high_freq_words]))

        naive_bayes(df, cv, nb_clf, label=label)
        # naive_bayes_train_test(df, cv, nb_clf, label=label)
        if counter >= 80:  # Comment this to go through rest of dataset
            break

    with open(cv_path, 'wb') as f:
        pickle.dump(cv, f)

    with open(pickle_path, 'wb') as f:
        pickle.dump(nb_clf, f)

    return nb_clf, cv


def get_k_highest_frequency_words(k=30):
    cv = load_pickle('pickles/Count_Vectorizer.pickle')
    nb_clf = load_pickle('pickles/NaiveBayes_Stars.pickle')

    total_word_count = {}
    dict1 = nb_clf.star1_words
    dict2 = nb_clf.star2_words
    dict3 = nb_clf.star3_words
    dict4 = nb_clf.star4_words
    dict5 = nb_clf.star5_words

    result = {key: dict1.get(key, 0) + dict2.get(key, 0)
              for key in set(dict1) | set(dict2)}
    result2 = {key: result.get(key, 0) + dict3.get(key, 0)
              for key in set(result) | set(dict3)}
    result3 = {key: result2.get(key, 0) + dict4.get(key, 0)
               for key in set(result2) | set(dict4)}
    result4 = {key: result3.get(key, 0) + dict5.get(key, 0)
               for key in set(result3) | set(dict5)}

    # print(dict1)
    # print(result)
    # print(result2)
    # print(result3)
    # print(result4)

    sorted_words = dict(sorted(result4.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_words)

    sorted_words_df = pd.DataFrame.from_dict(sorted_words,orient='index', columns=['COUNT'])
    sorted_words_df.index.name = 'WORD'
    sorted_words_df = sorted_words_df.reset_index()
    # sorted_words_df['WORD'] = sorted_words_df.index

    print(sorted_words_df.head(k))

    # import plotly.express as px
    # fig = px.bar(sorted_words_df.head(k), x="COUNT", y="WORD", title=f'Top {k} frequency words in data', orientation='h',
    #        width=1000, height=1000)
    # fig.update_layout(yaxis=dict(autorange="reversed"))
    # fig.show()

    return sorted_words_df.head(k)



def dropping_stars_1_to_5_model_train_val_result():
    train(label="stars", drop="stars1",
          pickle_path='pickles/NaiveBayes_Drop_Stars1.pickle', cv_path='pickles/Count_Vectorizer_Drop_Stars1.pickle')

    train(label="stars", drop="stars5",
          pickle_path='pickles/NaiveBayes_Drop_Stars5.pickle', cv_path='pickles/Count_Vectorizer_Drop_Stars5.pickle')

    train(label="stars", drop="stars2",
          pickle_path='pickles/NaiveBayes_Drop_Stars2.pickle', cv_path='pickles/Count_Vectorizer_Drop_Stars2.pickle')

    train(label="stars", drop="stars3",
          pickle_path='pickles/NaiveBayes_Drop_Stars3.pickle', cv_path='pickles/Count_Vectorizer_Drop_Stars3.pickle')

    train(label="stars", drop="stars4",
          pickle_path='pickles/NaiveBayes_Drop_Stars4.pickle', cv_path='pickles/Count_Vectorizer_Drop_Stars4.pickle')


    test(nb_clf=load_pickle('pickles/NaiveBayes_Drop_Stars1.pickle'), label="stars",
         cv=load_pickle('pickles/Count_Vectorizer_Drop_Stars1.pickle'), drop="stars1")

    test(nb_clf=load_pickle('pickles/NaiveBayes_Drop_Stars2.pickle'), label="stars",
         cv=load_pickle('pickles/Count_Vectorizer_Drop_Stars2.pickle'), drop="stars2")

    test(nb_clf=load_pickle('pickles/NaiveBayes_Drop_Stars3.pickle'), label="stars",
         cv=load_pickle('pickles/Count_Vectorizer_Drop_Stars3.pickle'), drop="stars3")

    test(nb_clf=load_pickle('pickles/NaiveBayes_Drop_Stars4.pickle'), label="stars",
         cv=load_pickle('pickles/Count_Vectorizer_Drop_Stars4.pickle'), drop="stars4")

    test(nb_clf=load_pickle('pickles/NaiveBayes_Drop_Stars5.pickle'), label="stars",
         cv=load_pickle('pickles/Count_Vectorizer_Drop_Stars5.pickle'), drop="stars5")



def dropping_k_highest_frequency_50_to_300_train_val_results():
    nb_clf300, cv300 = train_remove_k_highest_frequency(label="stars", k=300,
                                                        pickle_path="pickles/NaiveBayes_Stars_top_300_dropped.pickle",
                                                        cv_path='pickles/Count_Vectorizer_top_300_dropped.pickle')

    nb_clf200, cv200 = train_remove_k_highest_frequency(label="stars", k=200,
                                                        pickle_path="pickles/NaiveBayes_Stars_top_200_dropped.pickle",
                                                        cv_path='pickles/Count_Vectorizer_top_200_dropped.pickle')

    nb_clf100, cv100 = train_remove_k_highest_frequency(label="stars", k=100,
                                                        pickle_path="pickles/NaiveBayes_Stars_top_100_dropped.pickle",
                                                        cv_path='pickles/Count_Vectorizer_top_100_dropped.pickle')

    nb_clf50, cv50 = train_remove_k_highest_frequency(label="stars", k=50,
                                                      pickle_path="pickles/NaiveBayes_Stars_top_50_dropped.pickle",
                                                      cv_path='pickles/Count_Vectorizer_top_50_dropped.pickle')

    test(nb_clf=nb_clf300, label="stars", cv=cv300)
    test(nb_clf=nb_clf200, label="stars", cv=cv200)
    test(nb_clf=nb_clf100, label="stars", cv=cv100)
    test(nb_clf=nb_clf50, label="stars", cv=cv50)



if __name__ == '__main__':
    start_time = time.time()



    # train(label="stars", pickle_path="pickles/NaiveBayes_Stars_baseline.pickle",
    #       cv_path="pickles/Count_Vectorizer_baseline.pickle")
    test(nb_clf=load_pickle('pickles/NaiveBayes_Stars_top_15_dropped.pickle'), label="stars",
         cv=load_pickle("pickles/Count_Vectorizer_top_15_dropped.pickle"))

    # train(label="useful")
    # train(label="cool")
    # train(label="funny")
    # test(nb_clf=load_pickle('NaiveBayes_Stars.pickle'), label="stars")
    # test(nb_clf=load_pickle('NaiveBayes_Useful.pickle'), label="useful")
    # test(nb_clf=load_pickle('NaiveBayes_Cool.pickle'), label="cool")
    # test(nb_clf=load_pickle('NaiveBayes_Funny.pickle'), label="funny")

    # train_remove_k_highest_frequency(label="useful", k=15)
    # train_remove_k_highest_frequency(label="cool", k=15)
    # train_remove_k_highest_frequency(label="funny", k=15)
    # test(nb_clf=load_pickle('NaiveBayes_Stars_top_15_dropped.pickle'), label="stars")
    # test(nb_clf=load_pickle('NaiveBayes_Funny_top_15_dropped.pickle'), label="funny")

    # train_with_test(label="stars")

    print("END")
    print(f"--- {(time.time() - start_time):.2f} seconds ---")