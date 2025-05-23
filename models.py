import ipaddress
import warnings
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import pickle
from probablistic_models import *
from non_para_models import *

from sklearn import metrics
from IPython.core.display_functions import display
from IPython.core.display import HTML
from collections import OrderedDict

temp_dataset_name = "test.json"
temp_dataset_name_csv = "test.csv"
random_seed = 42

LABELS = ["stars", "useful", "funny", "cool"]
CLASS_LABEL = ["stars"]
REGRESSION_LABELS = ["useful", "funny", "cool"]

# -t test.json -c nb -v pickles/Count_Vectorizer.pickle -m pickles/NaiveBayes_Stars.pickle

def main():
    nn = "nn"
    dt = "dt"
    nb = "nb"

    parser = argparse.ArgumentParser(
        description='Implementation of sentiment analysis classifiers on YELP reviews dataset.')
    parser.add_argument('-t', '--testset', help='Path of heldout testset csv', required=True)
    parser.add_argument('-c', '--classifier', help=f'Name of classification method: \'{nn}\', \'{nb}\', '
                                                   f'\'{dt}\')', required=True)
    parser.add_argument('-v', '--vectorizer', help='Path of vectorizer to load', required=False)
    parser.add_argument('-m', '--model', help='Path of model to load', required=False)
    args = vars(parser.parse_args())

    classifer = args['classifier']

    # if args['model'] is not None:
    #     X_test, y_test_label, y_test_cat = load_withheld_testset_and_scale(args['testset'])
    #
    #     if task == "label":
    #         model_label_load_predict(model_name=args['classifier'], model=args['model'], feature_cols=X_test,
    #                                  y_labels=y_test_label)
    #     elif task == "attack_cat":
    #         model_attack_cat_load_predict(model_name=args['classifier'], model=args['model'], feature_cols=X_test,
    #                                       y_labels=y_test_cat)
    #     return


    if classifer == nb:
        if args['model'] is not None and args['vectorizer'] is not None:
            test(nb_clf=load_pickle(args['model']),
                 label="stars", cv=load_pickle(args['vectorizer']))
        else:
            train(label="stars", pickle_path="pickles/NaiveBayes_Stars_new.pickle",
                  cv_path="pickles/Count_Vectorizer_new.pickle")

            train(label="useful", pickle_path="pickles/NaiveBayes_Useful_new.pickle",
                  cv_path="pickles/Count_Vectorizer_new.pickle")

            train(label="cool", pickle_path="pickles/NaiveBayes_Cool_new.pickle",
                  cv_path="pickles/Count_Vectorizer_new.pickle")

            train(label="funny", pickle_path="pickles/NaiveBayes_Funny_new.pickle",
                  cv_path="pickles/Count_Vectorizer_new.pickle")

    elif classifer == nn:
        if args['model'] is not None and args['vectorizer'] is not None:
            print("NN Inference here")
        else:
            print("NN Train here, can skip")
    elif classifer == dt:
        if args['testset'] is not None:
            for label in LABELS:
                DT_class_predict(dataset_path=args["testset"], model_path=f"pickles/DT_class_model_{label}_tf_idf.pickle", label=label)
        else:
            train_DT_classifier("train_vectorized_tf_idf.csv")

if __name__ == '__main__':
    # create_train_val_test_set()
    start_time = time.time()
    main()
    print("END")
    print(f"--- {(time.time() - start_time):.2f} seconds ---")
