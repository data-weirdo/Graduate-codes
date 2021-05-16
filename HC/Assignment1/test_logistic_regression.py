import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def main():

    with open('./lr_parameter.pickle', 'rb') as f:
        lr_model = pickle.load(f)

    X_train = np.load('./X_train_logistic.npy')
    X_test = np.load('./X_test_logistic.npy')
    y_train = np.load('./y_train.npy')
    y_test = np.load('./y_test.npy')

    train_auroc = roc_auc_score(y_train, lr_model.predict_proba(X_train)[:,1])
    train_auprc = average_precision_score(y_train, lr_model.predict_proba(X_train)[:,1])
    test_auroc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:,1])
    test_auprc = average_precision_score(y_test, lr_model.predict_proba(X_test)[:,1])

    text_to_record = str(20213207) + '\n' + str(train_auroc) + '\n' \
        + str(train_auprc) + '\n' + str(test_auroc) + '\n' + str(test_auprc)

    with open('./20213207_logistic_regression.txt', 'w') as f:
        f.write(text_to_record)
    f.close()

if __name__ == '__main__':
    main()