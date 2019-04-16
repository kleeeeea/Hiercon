import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.hierarchical_att_model import HierAttNet
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import random
from gensim.models import Word2Vec
from tqdm import tqdms
# TODO:xx
DATA_ROOT = "./data/"


parser = argparse.ArgumentParser("")
parser.add_argument('arg1', nargs='?', default="cs", help="1st Positional arg")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epoches", type=int, default=5)
parser.add_argument("--log_interval", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--word_feature_size", type=int, default=4)
parser.add_argument("--sent_feature_size", type=int, default=3)
parser.add_argument("--num_bins", type=int, default=10)
parser.add_argument("--es_min_delta", type=float, default=0.0,
                    help="Early stopping's parameter: minimum change loss to qualify as an improvement")
parser.add_argument("--es_patience", type=int, default=5,
                    help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
parser.add_argument("--test_interval", type=int, default=1,
                    help="Number of epoches between testing phases")
parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
parser.add_argument("--saved_path", type=str, default=".")
args = parser.parse_args()
use_cuda = True
USE_TRAINING_METRICS = True


MAX_DOCS_PER_LABEL = '10000'

dataset = args.arg1
tokenized_text = '/disk/home/klee/data/{}_merged_tokenized'.format(dataset)

# embeddingpp
supersequence_path = tokenized_text + '_superspan_sequence.json'
model_save_path = supersequence_path + '_embedding.bin'
dictionary_path = tokenized_text + "_dictionary.bin"

# labels representation
label_namesFile = tokenized_text + '_class_ids.bin'
VvFile = tokenized_text + '_Vv.bin'
Vv_embedding_path = tokenized_text + '_Vv_embedding.bin'
basic_semanticsFile = tokenized_text + '_basic_semantics.bin'
path_semanticsFile = tokenized_text + '_path_semantics.bin'
childLabel2ParentLabelFile = tokenized_text + '_childid2parentid.bin'

# document representation
phrases2feature_vector_path = tokenized_text + '_phrases2feature_vector.bin'
superspan_HANsFile = tokenized_text + '_superspan_HANs.txt' + '{}'.format(MAX_DOCS_PER_LABEL)
superspan_HANs_labelsFile = tokenized_text + \
    '_superspan_HANs_labels.txt' + '{}'.format(MAX_DOCS_PER_LABEL)
ImportanceFeatureMatsFile = tokenized_text + \
    '_superspan_HANs_ImportanceFeatureMatsFile.bin' + '{}'.format(MAX_DOCS_PER_LABEL)

descriptor_HANsFile = tokenized_text + \
    '_descriptor_merged_label_HANs.json' + '{}'.format(MAX_DOCS_PER_LABEL)

# evaluation
training_inds_HANsFile = tokenized_text + \
    '_training_inds_HANsFile.bin' + '{}'.format(MAX_DOCS_PER_LABEL)
evaluationResultFile = tokenized_text + '_hiercon_evaluationResultFile.json'
evaluationResultFile_bin = tokenized_text + \
    '_test_metrics_hiercon.bin' + '{}'.format(MAX_DOCS_PER_LABEL)
evaluationResultFileFinal_bin = tokenized_text + \
    '_test_metrics_hiercon_final.bin' + '{}'.format(MAX_DOCS_PER_LABEL)

dataset2device = {
    'pubmed': 2,
    'cs': 0,
    'physmath': 0,
}

training_inds = []
training_inds = pickle.load(open(training_inds_HANsFile, 'rb'))

log_path = args.log_path + '_' + dataset


model_gensim = Word2Vec.load(model_save_path)

max_vocab = 800000

test_these_inds = None
test_this_label = None
get_concept_similarity = True

RANDOM_SEED = 123


def train(opt):
    if use_cuda:
        torch.cuda.manual_seed(RANDOM_SEED)
    else:
        torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": False}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": True,
                   "drop_last": False}

    mydataset = MyDataset(superspan_HANsFile, superspan_HANs_labelsFile, label_namesFile, ImportanceFeatureMatsFile,
                          model_gensim, max_vocab, training_inds, childLabel2ParentLabelFile, None, dataset, descriptor_HANsFile, VvFile)

    # training_generator = DataLoader(mydataset, **training_params)
    testing_inds = [i for i in range(len(mydataset.text_lines)) if i not in training_inds]
    if test_these_inds:
        testing_inds = test_these_inds
    test_set = MyDataset(superspan_HANsFile, superspan_HANs_labelsFile, label_namesFile, ImportanceFeatureMatsFile, model_gensim,
                         max_vocab, testing_inds, None, test_this_label, None, None, None, mydataset.max_length_sentences, mydataset.max_length_word)
    test_generator = DataLoader(test_set, **test_params)

    model = HierAttNet(opt.sent_feature_size, phrases2feature_vector_path, dictionary_path,
                       mydataset.max_length_sentences, mydataset.max_length_word,
                       model_save_path, Vv_embedding_path, path_semanticsFile, max_vocab, use_cuda, mydataset, opt.num_bins)
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    if use_cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    att_parameters = set(model.sent_att_net.parameters()) | set(model.word_att_net.parameters())
    optimizer = torch.optim.SGD([
        {'params': filter(lambda p: p.requires_grad, set(model.parameters()) - att_parameters)},
        {'params': filter(lambda p: p.requires_grad, att_parameters), 'lr': opt.lr * 1000},
    ], lr=opt.lr, momentum=opt.momentum)

    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)

    classind2size = Counter()
    topk2classind2errorSize = {
        'top 1': Counter(),
        'top 3': Counter(),
        'top 5': Counter(),
    }
    stop_training = False
    all_labels_set = set([l for l in mydataset.labels_list if '.' in l])
    sampled_labels_set = set()

    for epoch in range(opt.num_epoches):
        for iter, (features, ImportanceFeatureMat, labels, indexes, addtional_info) in tqdm(enumerate(training_generator)):
            if use_cuda:
                features = features.cuda()
                ImportanceFeatureMat = ImportanceFeatureMat.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            predictions, attn_score, similarity_w_attentions = model(
                features, ImportanceFeatureMat, get_concept_similarity)
            loss = criterion(predictions, labels)
            if not stop_training:
                loss.backward()
                optimizer.step()
            sampled_labels_set |= set(labels.cpu().numpy())

            if USE_TRAINING_METRICS:
                training_metrics = get_evaluation(labels.cpu().numpy(), predictions.data.cpu().numpy(), list_metrics=[
                                                  "accuracy", "top K accuracy", "top K classind2wrong_doc_ind", "top K tree score"], childLabel2ParentLabel=mydataset.childLabel2ParentLabel, labels_list=mydataset.labels_list)

                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, top K accuracy: {}, top K tree score: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss, training_metrics["top K accuracy"], training_metrics["top K tree score"]))

                writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Accuracy',
                                  training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)

                column_names = [model.dataset.labels_list.copy()
                                for _ in range(labels.shape[0])]

                if iter % opt.log_interval == 1:
                    if get_concept_similarity:
                        pickle.dump([pd.DataFrame(s, index=model.dataset.doc_tensor2doc(features[i]), columns=column_names[i]) for i, s in enumerate(
                            model.similarity_w_attentions)], open('log/model.similarity_w_attentions{}.bin'.format(iter), 'wb'))

                    pickle.dump(pd.DataFrame(model.bin_weight_history), open(
                        'log/model.bin_weight_history_{}.bin'.format(iter), 'wb'))
                    pickle.dump(pd.DataFrame(model.sent_att_net.context_weight_history, columns=['position', 'length', 'inTitle']),
                                open('log/model.sent_att_net.context_weight_history_{}.bin'.format(iter), 'wb'))
                    pickle.dump(pd.DataFrame(model.word_att_net.context_weight_history,
                                             columns=['meaningfulness', 'purity', 'targetness', 'completeness', 'nltk', 'spacy_np', 'spacy_entity', 'autophrase']),
                                open('log/model.word_att_net.context_weight_history_{}.bin'.format(iter), 'wb')
                                )

    model.eval()
    loss_ls = []
    te_label_ls = []
    te_pred_ls = []
    for iter, (te_feature, ImportanceFeatureMat, te_label, indexes, addtional_info) in tqdm(enumerate(test_generator), total=len(test_generator)):
        num_sample = len(te_label)
        if use_cuda:
            te_feature = te_feature.cuda()
            ImportanceFeatureMat = ImportanceFeatureMat.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            if test_these_inds or test_this_label:
                te_predictions, te_attn_score, similarity_w_attentions = model(
                    te_feature, ImportanceFeatureMat, get_concept_similarity=get_concept_similarity)
            else:
                te_predictions, te_attn_score, _ = model(te_feature, ImportanceFeatureMat)
        te_loss = criterion(te_predictions, te_label)
        loss_ls.append(te_loss * num_sample)
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())

        if test_these_inds or test_this_label:
            column_names = [model.dataset.labels_list.copy()
                            for _ in range(te_label.shape[0])]

            training_metrics = get_evaluation(te_label.cpu().numpy(), te_predictions.data.cpu().numpy(), list_metrics=[
                                              "accuracy", "top K accuracy", "top K classind2wrong_doc_ind", "top K tree score"], childLabel2ParentLabel=mydataset.childLabel2ParentLabel, labels_list=mydataset.labels_list)

            for classind, doc_ind_in_batchs in training_metrics["top K classind2wrong_doc_ind"]['top 1'].items():
                print('error for class: ', classind, mydataset.labels_list[classind])
                for (doc_ind, preds) in doc_ind_in_batchs:
                    try:
                        print('doc_ind', doc_ind, 'predicted: ', [
                              mydataset.labels_list[pred_classind] for pred_classind in preds], addtional_info[doc_ind])
                        for i, pred_classind in enumerate(preds):
                            column_names[doc_ind][pred_classind] += "@{}".format(i)
                        print(mydataset.doc_tensor2doc(
                            te_feature[doc_ind]), 'tensor index:', indexes[doc_ind])
                        sim_save_path = 'log/model.similarity_w_attentions_docindex_{}.bin'.format(indexes.numpy()[
                                                                                                   doc_ind])
                        pickle.dump(pd.DataFrame(similarity_w_attentions[doc_ind], index=model.dataset.doc_tensor2doc(
                            te_feature[doc_ind]), columns=column_names[doc_ind]), open(sim_save_path, 'wb'))
                    except Exception as e:
                        import ipdb
                        ipdb.set_trace()
                        raise e
            continue

        if iter % 10 == 1:
            print('test iter {}/{}'.format(iter, len(test_generator)))

            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=[
                                          "accuracy", "top K accuracy", "top K tree score", "top K accuracy by class", "confusion_matrix"], childLabel2ParentLabel=mydataset.childLabel2ParentLabel, labels_list=mydataset.labels_list)

            with open(evaluationResultFile, 'w') as my_file:
                # print(str(test_metrics), file=my_file)

            pickle.dump(test_metrics, open(evaluationResultFile_bin, 'wb'))
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            writer.add_scalar('Test/Loss', te_loss, epoch)
            writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(
                    epoch, te_loss))
                break

    pickle.dump(test_metrics, open(evaluationResultFileFinal_bin, 'wb'))


if __name__ == '__main__':
    if use_cuda:
        torch.cuda.set_device(dataset2device[dataset])

    train(args)
