import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, confusion_matrix
LABELS = ["stars", "useful", "funny", "cool"]
CLASS_LABEL = ["stars"]
REGRESSION_LABELS = ["useful", "funny", "cool"]

def train_DT_classifier(trainset):
    HYPERPARAMS = {"max_depth": [2,4,6,8,10], "criterion": ["gini", "entropy"], "min_samples_split": [2,3,4], "random_state": [42]}

    train = pd.read_csv(trainset)
    
    train.dropna(inplace=True)
    print(train.shape)
    # sample 100k record subset for faster hyperparam tuning
    train_subset = train.sample(n=100000, random_state=42)
    for target in CLASS_LABEL:       
        # Hyperparamter tuning
        print(f"Classification Target: {target}")
        gs = GridSearchCV(estimator=DecisionTreeClassifier(), scoring="f1_micro", n_jobs=-1, param_grid=HYPERPARAMS)
        gs.fit(X=train_subset.drop(LABELS, axis=1), y=train_subset[target])
        print(f"Optimal Hyperparameters: {gs.best_params_}")
        print(f"Best Score: {gs.best_score_}")

        # Final training
        print(f"Training Decision Tree Classifier for label: {target}")
        clf = DecisionTreeClassifier(**gs.best_params_)
        clf = clf.fit(X=train.drop(LABELS, axis=1), y=train[target])
        print(f"Saving Decision Tree Classifier model")
        pickle.dump(clf, open(f'pickles/DT_class_model_{target}.pickle', 'wb'))

# def train_DT_regressor():
#     HYPERPARAMS = {"max_depth": [2,4,6], "criterion": ["gini", "log_loss", "entropy"], "min_samples_split": [2,3,4], "random_state": [42]}

#     train = pd.read_csv("train_vectorized_tf_idf.csv")
#     train.dropna(inplace=True)

#     # Create 50,000 sample subset for Hyperparameter tuning using Grid Search
#     train_subset = train.sample(n=100000, random_state=42)

#     # Hyperparamter tuning
#     for reg_label in REGRESSION_LABELS:
#         print(f"Regression Target: {train_subset[0]}")
#         gs = GridSearchCV(estimator=DecisionTreeRegressor(), scoring="neg_mean_squared_error", n_jobs=-1, param_grid=HYPERPARAMS)
#         gs.fit(gs.fit(X=train_subset.drop(LABELS, axis=1), y=train_subset[reg_label]))
#         print(f"Optimal Hyperparameters: {gs.best_params}")
#         print(f"Best Score: {gs.best_score_}")

#         # Final training
#         print(f"Training Decision Tree Regressor for label: {reg_label}")
#         clf = DecisionTreeRegressor(**gs.best_params_)
#         clf = clf.fit(X=train_subset.drop(LABELS, axis=1), y=train_subset[reg_label])
#         print(f"Saving Decision Tree Regressor model: {reg_label}")
#         pickle.dump(clf, open(f'DT_reg_model_{reg_label}.pickle', 'wb'))   

def DT_class_predict(dataset_path: str, model_path: str, label: str):
    test = pd.read_csv(dataset_path)
    test.dropna(inplace=True)
    X_test = test.drop(LABELS, axis=1)
    print("Classifier Predictions")
    print(f"Predicting Label: {label}")
    model = open(model_path,"rb")
    clf : DecisionTreeClassifier = pickle.load(model)
    y_pred = clf.predict(X_test)
    print(f"Classification Report:\n{classification_report(y_pred=y_pred, y_true=(test[label].reset_index(drop=True)))}")
    print("\n")

# def DT_reg_predict(dataset_path: str, model_path: str):
#     test = pd.read_csv(dataset_path)
#     test.dropna(inplace=True)
#     X_test = test.drop(LABELS, axis=1)
#     print("Regression Predictions")
#     for label in REGRESSION_LABELS:
#         print(f"Predicting Label: {label}")
#         model = open(model_path,"rb")
#         clf : DecisionTreeRegressor = pickle.load(model)
#         y_pred = clf.predict(X_test)
#         print(f"Mean Squared Error: {mean_squared_error(y_pred=y_pred, y_true=test[label])}")
#         print(f"Root Mean Squared Error: {mean_squared_error(y_pred=y_pred, y_true=test[label], squared=False)}")
#         print(f"Mean Absolute Error: {mean_absolute_error(y_pred=y_pred, y_true=test[label])}")
#     print("\n")

if __name__ == "__main__":
    # train_DT_classifier()

    for label in CLASS_LABEL:
        DT_class_predict("hand_anno_vec.csv", f"DT_class_model_{label}_tf_idf.pickle", label)

    # train_DT_regressor()
    # DT_reg_predict("val_vectorized_tf_idf.csv", "DT_reg_model.pickle")
