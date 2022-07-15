import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


def logger(info):
    epoch = info['epoch']
    train_loss, test_acc, test_std = info['train_loss'], info[
        'test_acc'], info['test_std']
    print(
        f'{epoch:03d}: Train Loss: {train_loss:.3f}, Test Acc: {test_acc:.4f} '
        f'± {test_std:.4f}')


def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(),
                                      params,
                                      cv=5,
                                      scoring='accuracy',
                                      verbose=0)
        else:
            classifier = SVC(C=5)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies), np.std(accuracies)


def evaluate_embedding(embeddings, labels, search=False):
    x, y = np.array(embeddings), np.array(labels)

    svc_acc, svc_std = svc_classify(x, y, search=search)
    # print(f'{svc_acc:.4f} ± {svc_std:.4f}')
    return svc_acc, svc_std
