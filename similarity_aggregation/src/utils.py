import numpy as np
from sklearn import metrics
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import torch.nn.functional as F
import sys
import csv
from collections import defaultdict, Counter
csv.field_size_limit(sys.maxsize)


def top_k_accuracy_score(y_true, y_prob, k=5, normalize=True):
    """Top k Accuracy classification score.
    For multiclass classification tasks, this metric returns the
    number of times that the correct class was among the top k classes
    predicted.
    Parameters
    ----------
    y_true : 1d array-like, or class indicator array / sparse matrix
        shape num_samples or [num_samples, num_classes]
        Ground truth (correct) classes.
    y_pred : array-like, shape [num_samples, num_classes]
        For each sample, each row represents the
        likelihood of each possible class.
        The number of columns must be at least as large as the set of possible
        classes.
    k : int, optional (default=5) predictions are counted as correct if
        probability of correct class is in the top k classes.
    normalize : bool, optional (default=True)
        If ``False``, return the number of top k correctly classified samples.
        Otherwise, return the fraction of top k correctly classified samples.
    Returns
    -------
    score : float
        If ``normalize == True``, return the proportion of top k correctly
        classified samples, (float), else it returns the number of top k
        correctly classified samples (int.)
        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.
    See also
    --------
    accuracy_score
    Notes
    -----
    If k = 1, the result will be the same as the accuracy_score (though see
    note below). If k is the same as the number of classes, this score will be
    perfect and meaningless.
    In cases where two or more classes are assigned equal likelihood, the
    result may be incorrect if one of those classes falls at the threshold, as
    one class must be chosen to be the nth class and the class chosen may not
    be the correct one.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import top_k_accuracy_score
    >>> y_pred = np.array([[0.1, 0.3, 0.4, 0.2],
    ...                    [0.4, 0.3, 0.2, 0.1],
    ...                    [0.2, 0.3, 0.4, 0.1],
    ...                    [0.8, 0.1, 0.025, 0.075]])
    >>> y_true = np.array([2, 2, 2, 1])
    >>> top_k_accuracy_score(y_true, y_pred, k=1)
    0.5
    >>> top_k_accuracy_score(y_true, y_pred, k=2)
    0.75
    >>> top_k_accuracy_score(y_true, y_pred, k=3)
    1.0
    >>> top_k_accuracy_score(y_true, y_pred, k=2, normalize=False)
    3
    """
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    num_obs, num_labels = y_prob.shape
    counter = 0
    argsorted = np.argsort(-y_prob, axis=1)
    parentid2wrong_docid_preds = defaultdict(list)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, :k]:
            counter += 1
        else:
            parentid2wrong_docid_preds[y_true[i]].append((i, argsorted[i, :5]))
            pass

    if normalize:
        counter = counter / num_obs

    return counter, parentid2wrong_docid_preds


def top_k_tree_score(y_true, y_prob, childLabel2ParentLabel, labels_list, k=5, normalize=True):
    nonleaves = set(childLabel2ParentLabel.values())

    def get_closest_nonleaves_lave(ind):
        try:
            label = labels_list[ind]
            return label if label in nonleaves else childLabel2ParentLabel.get(label)
        except Exception as e:
            return label

    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    num_obs, num_labels = y_prob.shape
    counter = 0
    argsorted = np.argsort(-y_prob, axis=1)
    parentid2wrong_docid_preds = defaultdict(list)
    for i in range(num_obs):
        # leave nodes are normalized to its parent
        parents_pred = [get_closest_nonleaves_lave(ind) for ind in argsorted[i, :k]]
        if get_closest_nonleaves_lave(y_true[i]) in parents_pred:
            counter += 1
        else:
            parentid2wrong_docid_preds[childLabel2ParentLabel[labels_list[y_true[i]]]].append(
                (i, parents_pred))
            pass

    if normalize:
        counter = counter / num_obs

    return counter, parentid2wrong_docid_preds


def get_evaluation(y_true, y_prob, list_metrics, childLabel2ParentLabel, labels_list):
    # get per class accuracy & error case
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if "top K tree score" in list_metrics:
        output['top K tree score'] = {
            'top 1': top_k_tree_score(y_true, y_prob, childLabel2ParentLabel, labels_list, 1)[0],
            'top 3': top_k_tree_score(y_true, y_prob, childLabel2ParentLabel, labels_list, 3)[0],
            'top 5': top_k_tree_score(y_true, y_prob, childLabel2ParentLabel, labels_list, 5)[0],
        }
    if "top K classind2wrong_doc_ind" in list_metrics:
        output['top K classind2wrong_doc_ind'] = {
            'top 1': top_k_accuracy_score(y_true, y_prob, 1)[1],
            'top 3': top_k_accuracy_score(y_true, y_prob, 3)[1],
            'top 5': top_k_accuracy_score(y_true, y_prob, 5)[1],
        }
    if "top K accuracy" in list_metrics:
        output['top K accuracy'] = {
            'top 1': top_k_accuracy_score(y_true, y_prob, 1)[0],
            'top 3': top_k_accuracy_score(y_true, y_prob, 3)[0],
            'top 5': top_k_accuracy_score(y_true, y_prob, 5)[0],
        }

    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))

    if 'top K accuracy by class' in list_metrics:
        y_true_index2size = Counter(list(y_true))

        topk2y_true_index2wrong_ratio = {}
        output['top K classind2wrong_doc_ind'] = {
            'top 1': top_k_accuracy_score(y_true, y_prob, 1)[1],
            'top 3': top_k_accuracy_score(y_true, y_prob, 3)[1],
            'top 5': top_k_accuracy_score(y_true, y_prob, 5)[1],
        }

        for topN, classind2wrong_doc_ind in output["top K classind2wrong_doc_ind"].items():
            y_true_index2wrong_ratio = {}
            counter_classind2wrong_doc_ind = Counter(
                {ind: len(wrong_doc_inds) for ind, wrong_doc_inds in classind2wrong_doc_ind.items()})
            # test_metrics["top K classind2wrong_doc_ind"]
            for y_true_indexs, size in y_true_index2size.items():
                y_true_index2wrong_ratio[labels_list[y_true_indexs]] = 1 - \
                    counter_classind2wrong_doc_ind.get(y_true_indexs, 0) / size
            topk2y_true_index2wrong_ratio[topN] = y_true_index2wrong_ratio

        output['top K accuracy by class'] = topk2y_true_index2wrong_ratio

    return output


def batch_matrix_mul(input, weight, bias=False):
    feature_list = []
    # need to handle one dimension
    try:
        for feature in input:
            feature = torch.mm(feature, weight)
            if isinstance(bias, torch.nn.parameter.Parameter):
                feature = feature + bias
                # feature + bias.expand(feature.size()[0], bias.size()[1])
                # feature + bias
            feature = torch.tanh(feature).unsqueeze(0)
            feature_list.append(feature)
        result = torch.cat(feature_list, 0).squeeze(2)
    except Exception as e:
        import ipdb
        ipdb.set_trace()
        raise e

    return result


def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def zero_gating_softmax_normalize(output):
    return F.normalize(output * F.softmax(output, dim=1), dim=1)


def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in enumerate(reader):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]


if __name__ == "__main__":
    word, sent = get_max_lengths("../data/test.csv")
    print(word)
    print(sent)
