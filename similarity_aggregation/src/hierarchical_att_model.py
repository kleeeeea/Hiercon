import pickle
import numpy as np
import torch
import torch.nn as nn
from .sent_att_model import SentAttNet
from .word_att_model import WordAttNet
from .utils import batch_matrix_mul, element_wise_mul
import torch.nn.functional as F
import pandas as pd

BIN_START = -0.5


def create_embedding(mat):
    mat = torch.from_numpy(mat)
    embedding = nn.Embedding(
        num_embeddings=mat.shape[0], embedding_dim=mat.shape[1]).from_pretrained(mat)
    embedding.weight.requires_grad = False
    return embedding


class HierAttNet(nn.Module):
    def __init__(self, sent_feature_size, feature_path, dict,
                 max_sent_length, max_word_length,
                 model_save_path, Vv_embedding_path, phi_vsFile, max_vocab, use_cuda, dataset, num_bins_woexactmatch):
        super(HierAttNet, self).__init__()
        self.dataset = dataset
        self.use_cuda = use_cuda
        self.sent_feature_size = sent_feature_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(feature_path, dict, max_vocab, use_cuda, dataset)
        self.sent_att_net = SentAttNet(sent_feature_size)

        self.model_gensim = self.dataset.model_gensim
        self.model_gensim.most_similar(self.model_gensim.wv.index2word[0])
        feature = self.model_gensim.wv.vectors_norm[:max_vocab]
        unknown_word = np.zeros((1, self.model_gensim.vector_size))
        feature = np.concatenate([unknown_word, feature], axis=0).astype(np.float)
        self.embedding = create_embedding(feature)

        self.bin_midpoint = np.linspace(BIN_START, .99, 15).tolist()
        # adding extra bin for exact match
        bins_weight = [BIN_START]
        for i in range(len(self.bin_midpoint)-1):
            bins_weight.append((self.bin_midpoint[i] + self.bin_midpoint[i + 1])/2)
        bins_weight.append(1)
        bins_weight_difference = np.diff(bins_weight).tolist()
        bins_weight_difference.insert(0, 0)
        bins_weight_difference = torch.from_numpy(np.array(bins_weight_difference))
        self.bin_weight_difference = nn.Parameter(torch.Tensor(len(bins_weight_difference)))
        self.bin_weight_difference.data.copy_(bins_weight_difference)
        self.bin_weight_difference_start = nn.Parameter(torch.Tensor([bins_weight[0]]))

        try:
            Vv_embedding = pickle.load(open(Vv_embedding_path, 'rb'))
            self.Vv_embeddingT = torch.from_numpy(Vv_embedding.T.astype(np.float64))
            if use_cuda:
                self.Vv_embeddingT = self.Vv_embeddingT.cuda()
            self.Vv_embeddingT.requires_grad = False
        except Exception as e:
            import ipdb
            ipdb.set_trace()
            raise e

        phi_vs = pickle.load(open(phi_vsFile, 'rb'))
        self.phi_vs = F.normalize(torch.from_numpy(phi_vs.astype(np.float64)), 1)
        if use_cuda:
            self.phi_vs = self.phi_vs.cuda()
        self.phi_vs.requires_grad = False

        self.bin_weight_history = []

    def forward(self, input, ImportanceFeatureMat, get_concept_similarity=False):
        batch_size = input.size(0)
        word_attn_score = torch.zeros(self.max_sent_length, batch_size, self.max_word_length)
        if self.use_cuda:
            word_attn_score = word_attn_score.cuda()
        output_list = []

        for idx, i in enumerate(input.permute(1, 0, 2)):
            try:
                output, word_attn_score[idx] = self.word_att_net(i)
            except Exception as e:
                import ipdb
                ipdb.set_trace()
                raise e
            output_list.append(output)

        word_attn_score = word_attn_score.permute(1, 0, 2)
        output = torch.cat(output_list, 0)

        sent_attn_score = self.sent_att_net(ImportanceFeatureMat)
        attn_score = word_attn_score.permute(2, 0, 1) * sent_attn_score
        attn_score = attn_score.permute(1, 2, 0)
        attn_score = attn_score.contiguous().view(batch_size, -1)
        doc_index = input.view(batch_size, -1)

        final_score = self.compute_score(doc_index, attn_score, get_concept_similarity)
        return final_score, attn_score, self.similarity_w_attentions

    def compute_score(self, doc_index, attn_score, get_concept_similarity):
        bin_weight_computed = self.bin_weight_difference_start + \
            torch.cumsum(F.relu(self.bin_weight_difference), 0)
        self.bin_weight_history.append(bin_weight_computed.data.cpu().numpy())
        batch_size = doc_index.size(0)
        Nd_len = doc_index.size(1)
        self.similarity_w_attentions = [
            np.zeros((Nd_len, self.phi_vs.shape[0]), np.float) for _ in range(batch_size)]

        node_num = self.phi_vs.size(0)
        final_score = torch.zeros(batch_size, node_num)
        if self.use_cuda:
            final_score = final_score.cuda()
        for i in range(batch_size):
            similarity_mat = self.embedding(doc_index[i]).mm(self.Vv_embeddingT)

            similarity_mat_digitized = np.digitize(similarity_mat.cpu().numpy(), self.bin_midpoint)
            similarity_mat_digitized = torch.from_numpy(similarity_mat_digitized)
            if self.use_cuda:
                similarity_mat_digitized = similarity_mat_digitized.cuda()

            similarity_mat_digitized_one_hot = torch.zeros(
                similarity_mat_digitized.shape[0], similarity_mat_digitized.shape[1], 16)
            if self.use_cuda:
                similarity_mat_digitized_one_hot = similarity_mat_digitized_one_hot.cuda()

            similarity_mat_digitized_one_hot.scatter_(2, similarity_mat_digitized.unsqueeze(2), 1)

            Vd = self.dataset.doc_tensor2doc(doc_index[i])

            raw_word_attns = [i for i in list(zip(Vd, attn_score[i].data.cpu().numpy()))]
            word_attns = [i for i in list(zip(Vd, attn_score[i].data.cpu().numpy())) if i[0]]

            # visualize similarity attention to that dataset
            for j in range(node_num):
                try:
                    def get_similarity_by_concept():
                        # get concept wise interaction matrix
                        similarity_by_concept = batch_matrix_mul(
                            similarity_mat_digitized_one_hot, bin_weight_computed.unsqueeze(1)) * self.phi_vs[j]
                        similarity_by_concept = similarity_by_concept.sum(1)

                        return torch.sum(attn_score[i] * similarity_by_concept), attn_score[i] * similarity_by_concept

                    def get_similarity_to_train_bin_weight_difference():
                        outprod_score = torch.ger(attn_score[i], self.phi_vs[j])

                        similarity_histogram = (outprod_score.unsqueeze(
                            2) * similarity_mat_digitized_one_hot).sum((0, 1))
                        score = torch.sum(similarity_histogram * bin_weight_computed)

                        return score

                    final_score[i, j] = get_similarity_to_train_bin_weight_difference()
                    if get_concept_similarity:
                        _, similarity_by_concept = get_similarity_by_concept()
                        self.similarity_w_attentions[i][:,
                                                        j] = similarity_by_concept.data.cpu().numpy()

                except Exception as e:
                    import ipdb
                    ipdb.set_trace()
                    pass

        return final_score
