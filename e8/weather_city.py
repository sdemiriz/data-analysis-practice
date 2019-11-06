from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
import pandas as pd
import sys

def main():
    labelled = pd.read_csv(sys.argv[1])
    unlabelled = pd.read_csv(sys.argv[2])

    labels = labelled['city']
    features = labelled.loc[:, labelled.columns != 'city']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)

    mlp_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver= 'lbfgs', activation= 'logistic')
    )
    mlp_model.fit(X_train, y_train)
    print("mlp_model score = ", mlp_model.score(X_test, y_test))

    # Produced worse results than MLP

    # bayesian_model = make_pipeline(
    #     StandardScaler(),
    #     GaussianNB()
    # )
    # bayesian_model.fit(X_train, y_train)
    # print("bayesian_model score = ", bayesian_model.score(X_test, y_test))

    # knn_model = make_pipeline(
    #     StandardScaler(),
    #     KNeighborsClassifier(n_neighbors=5)
    # )
    # knn_model.fit(X_train, y_train)
    # print("knn_model score = ", knn_model.score(X_test, y_test))

    prediction = mlp_model.predict(unlabelled.iloc[:, unlabelled.columns != 'city'])
    pd.Series(prediction).to_csv(sys.argv[3], index=False, header=False)


if __name__ == '__main__':
    main()
