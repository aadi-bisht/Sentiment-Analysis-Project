import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import argparse

OUTPUT_PATH = "output.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Filepath of the dataset to vectorize/transform")
    ARGS = parser.parse_args()

    data = pd.read_json(ARGS.filepath, lines=True)
    data.dropna(inplace=True)

    vectorizer = CountVectorizer(max_features=100, ngram_range=(1,2))
    transformer = TfidfTransformer()

    count_vectorizer = vectorizer.fit(data["text"])
    transformer = transformer.fit(count_vectorizer)

    counts = vectorizer.transform(data["text"])
    tfidf = transformer.transform(counts)
    labels = transformer.get_feature_names_out()
    print(labels)
    print(f"Saving to {OUTPUT_PATH}")
    data.drop(["text"], axis=1, inplace=True)
    data = pd.concat([data, pd.DataFrame(tfidf.toarray(), columns=labels)], axis=1)
    data.to_json(OUTPUT_PATH, lines=True)
if __name__ == "__main__":
    main()