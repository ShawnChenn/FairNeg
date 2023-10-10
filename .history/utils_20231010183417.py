'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
import random
import os
import pickle
import itertools
from torch.utils.data.sampler import Sampler

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    print(path)
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False

def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value >= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value <= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag

class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne_pure(self, users, pos, neg):
        loss, reg_loss, fair_reg_loss = self.model.bpr_loss(users, pos, neg)

        reg_loss = (reg_loss) * self.weight_decay
        # loss = loss + reg_loss
        loss = loss + reg_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

    def stageOne(self, users, pos, neg, datapath, pklname, key_genre, pos_cpu):
        loss, reg_loss, fair_reg_loss = self.model.bpr_loss(users, pos, neg, datapath, pklname, key_genre, pos_cpu)

        reg_loss = (reg_loss) * self.weight_decay
        loss = loss + reg_loss
        # loss = loss + reg_loss + 0.01 * fair_reg_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

    def stageOne_reg(self, users, pos, neg, datapath, pklname, key_genre, pos_cpu):
        loss, reg_loss, fair_reg_loss = self.model.bpr_loss_reg(users, pos, neg, datapath, pklname, key_genre, pos_cpu)

        reg_loss = (reg_loss) * self.weight_decay
        # loss = loss + reg_loss
        loss = loss + reg_loss + 0.3 * fair_reg_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()



def UniformSample_original(datapath, dataset, key_genre, item_dict, neg_prob, neg_ratio=5):
    dataset : BasicDataset
    allPos = dataset.allPos
    if not sample_ext:
    # if not sample_ext:
        print('cpp uniform negative sampling...')
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        # pos neg distribution the same
        S = UniformSample_python(datapath, dataset, key_genre, item_dict, neg_prob, neg_ratio)
        # pos neg distribution differs
        # S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)

    x = [i for i in range(dataset.m_items)]

    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            # negitem = np.random.choice(a=x, size=1, replace=True, p=prob)[0]
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

class AdamOptimizer(object):

    def __init__(self, params):
        """
        params: each class's sampling probability
        grads: each class's deviation degree from the average performance
        """
        self.params = params
        self.momentum = None

    @torch.no_grad()
    def step(self, lr=0.2, params=None, grads=None):
        # # non-momentum sgd update rule
        # out_params = []
        # for (p, g) in zip(params, grads):
        #     d_p = p
        #     new_p = d_p - g * lr
        #     out_params.append(new_p)
        # return out_params

        # momentum sgd update
        out_params = []
        if self.momentum is None:
            self.momentum = [None] * len(params)

        for (p, g) in zip(params, grads):
            index = params.index(p)
            if self.momentum[index] is None:
                new_buf = g
            else:
                buf = self.momentum[index]
                new_buf = buf * 0.9 + g * 0.1

            self.momentum[index] = g
            new_p = p - new_buf * lr
            if new_p > 1:
                new_p = 1
            if new_p < 0:
                new_p = 0
            out_params.append(new_p)
        return out_params

class hardNeg_Sample_python():
    def __init__(self, dataset, datapath, pklname, model, prob, key_genre):
        self.rec_model = model
        self.prob = prob
        self.outer_optimizer = AdamOptimizer(self.prob)
        self.item_idd_category = pickle.load(open(datapath + '/' + pklname, 'rb'))

        allPos = dataset.allPos
        self.train_pairs = []
        for i in range(dataset.n_users):
            for j in allPos[i]:
                self.train_pairs.append((i, j))

        self.zp_attribute = []

        for p in np.array(self.train_pairs)[:, 1]:
            gs = self.item_idd_category[p]
            for g in gs:
                if g in key_genre:
                    self.zp_attribute.append(key_genre.index(g))
                    break

        self.z_item = list(set(self.zp_attribute))
        self.zp_index = {}
        self.zp_mask = {}
        self.zp_len = {}

        for tmp_z in self.z_item:
            self.zp_mask[tmp_z] = torch.Tensor([1 if i == tmp_z else 0 for i in self.zp_attribute])

        for tmp_z in self.z_item:
            self.zp_index[tmp_z] = (self.zp_mask[tmp_z] == 1).nonzero().squeeze()

        for tmp_z in self.z_item:
            self.zp_len[tmp_z] = len(self.zp_index[tmp_z])

    def softmax_with_temperature(self, X, T=0.4):
        X = X / T
        max_prob = np.max(X, axis=0)
        X -= max_prob
        exp_x = np.exp(X, X)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def ranking_aware_prob(self, rank_order_list):
        total_len = len(rank_order_list)
        prob = []
        for i in rank_order_list:
            prob_i = (4 * total_len) / pow(total_len - 1, 2) * pow((i / total_len), 3)
            prob.append(prob_i)

        prob[prob.index(max(prob))] += 1 - sum(prob)

        return prob

    def cal_fair_prob(self, Recmodel):
        with torch.no_grad():
            users_gpu = torch.Tensor(np.array(self.train_pairs)[:, 0]).long().cuda()
            pos_gpu = torch.Tensor(np.array(self.train_pairs)[:, 1]).long().cuda()

            loss = Recmodel.get_pos_logit(users_gpu, pos_gpu)
            yhat_yzp = {}

            for tmp_z in self.z_item:
                yhat_yzp[tmp_z] = float(torch.sum(loss[self.zp_index[tmp_z]])) / self.zp_len[tmp_z]

            yhat_yzp_items = list(yhat_yzp.values())
            yhat_yzp_avg = sum(yhat_yzp_items) / len(yhat_yzp_items)
            cur_grads = np.array([y - yhat_yzp_avg for y in yhat_yzp_items])
            self.prob = self.outer_optimizer.step(params=self.prob, grads=cur_grads)
            self.prob[-1] += 1 - sum(self.prob)
            print('fair group prob calc finished', self.prob)
            return self.prob

    def fair_aware_prob(self, ss_group_prob, candidate_items, key_genre):
        candidate_items_categroy = np.array([key_genre.index([j for j in self.item_idd_category[i] if j in key_genre][0]) \
                                    for i in candidate_items])

        prob = np.zeros(candidate_items_categroy.shape[0], dtype=np.float32)
        for k in range(len(key_genre)):
            key_genre_cnt = np.sum(candidate_items_categroy == k)
            prob[candidate_items_categroy == k] = float(ss_group_prob[k]) / float(key_genre_cnt)
        prob[-1] += float(1) - np.sum(prob)
        return prob

    def sample_neg(self, dataset, datapath, key_genre, pklname, Recmodel, balance_ratio):
        dataset: BasicDataset
        Recmodel = Recmodel.eval()
        users = [i for i in range(dataset.n_users)]
        items = [j for j in range(dataset.m_items)]

        allPos = dataset.allPos
        S = []

        start = time()

        sensitive_group_prob = self.cal_fair_prob(Recmodel)

        with torch.no_grad():
            users_gpu = torch.Tensor(np.array(users)).long().cuda()
            # 3845 * 2487
            ratings = Recmodel.getUsersRating(users_gpu).cpu().numpy()

            for i, user in enumerate(users):
                start = time()
                posForUser = allPos[user]
                if len(posForUser) == 0:
                    continue

                candidate_items = list(set(items) - set(posForUser))

                neg_scores = ratings[user][candidate_items]
                hard_aware_neg_prob = self.softmax_with_temperature(X=neg_scores)
                fair_aware_neg_prob = self.fair_aware_prob(sensitive_group_prob, candidate_items, key_genre)

                assert len(hard_aware_neg_prob) == len(fair_aware_neg_prob)

                total_prob = balance_ratio * np.array(hard_aware_neg_prob) + (1 - balance_ratio) * np.array(fair_aware_neg_prob)
                total_prob[-1] += 1 - sum(total_prob)

                for j in range(len(posForUser)):
                    pos_index = j
                    pos_item = posForUser[pos_index]

                    neg_item = int(np.random.choice(candidate_items, size=1, p=total_prob)[0])
                    S.append([user, pos_item, neg_item])

            end = time()
            total = end - start
            print('sampling time', total)
            return np.array(S)
        end = time()

        total = end - start
        print('sampling time', total)


class UniformSample_python_v1():
    def __init__(self, neg_prob):
        # 'Sci-Fi', "Horror", 'Romance', 'Crime'
        # ['Grocery', 'Toy']
        self.prob = neg_prob
        self.outer_optimizer = AdamOptimizer(self.prob)
        # self.upper_bound = [0.31, 0.24, 0.36, 0.09]

    # def softmax_with_temperature(self, yhat_yz_p, T=1, copy=True):
    #     yhat_yzp_items = list(yhat_yz_p.values())
    #     yhat_yzp_avg = sum(yhat_yzp_items) / len(yhat_yzp_items)
    #     X = np.array([y - yhat_yzp_avg for y in yhat_yzp_items])
    #     print('X', X)
    #     X = X / T
    #     if copy:
    #         X = np.copy(X)
    #     max_prob = np.max(X, axis=0)
    #     X -= max_prob
    #     np.exp(X, X)
    #     sum_prob = np.sum(X, axis=0)
    #     X /= sum_prob
    #     return X

    def Update_UniformSample_python(self, dataset, datapath, key_genre, pklname, item_dict, Recmodel, pres):
            dataset: BasicDataset
            Recmodel = Recmodel.eval()
            users = [i for i in range(dataset.n_users)]
            items = [i for i in range(dataset.m_items)]

            allPos = dataset.allPos
            S = []

            start = time()

            if pres is not None:
                zp_attribute = []
                zn_attribute = []
                f1 = pickle.load(open(datapath + '/' + pklname, 'rb'))
                for p in np.array(pres)[:, 1]:
                    gs = f1[p]
                    for g in gs:
                        if g in key_genre:
                            zp_attribute.append(key_genre.index(g))
                            break

                for n in np.array(pres)[:, 2]:
                    gs = f1[n]
                    for g in gs:
                        if g in key_genre:
                            zn_attribute.append(key_genre.index(g))
                            break

                z_item = list(set(zp_attribute))
                zp_index = {}
                zn_index = {}
                zp_mask = {}
                zn_mask = {}
                yhat_yzp = {}
                yhat_yzn = {}
                zp_len = {}
                zn_len = {}

                for tmp_z in z_item:
                    zp_mask[tmp_z] = torch.Tensor([1 if i == tmp_z else 0 for i in zp_attribute])
                    zn_mask[tmp_z] = torch.Tensor([1 if i == tmp_z else 0 for i in zn_attribute])

                for tmp_z in z_item:
                    zp_index[tmp_z] = (zp_mask[tmp_z] == 1).nonzero().squeeze()
                    zn_index[tmp_z] = (zn_mask[tmp_z] == 1).nonzero().squeeze()

                for tmp_z in z_item:
                    zp_len[tmp_z] = len(zp_index[tmp_z])
                    zn_len[tmp_z] = len(zn_index[tmp_z])

                with torch.no_grad():
                    users_gpu = torch.Tensor(np.array(pres)[:, 0]).long().cuda()
                    pos_gpu = torch.Tensor(np.array(pres)[:, 1]).long().cuda()
                    neg_gpu = torch.Tensor(np.array(pres)[:, 2]).long().cuda()

                    loss, _ = Recmodel.getlogit(users_gpu, pos_gpu, neg_gpu)
                    for tmp_z in z_item:
                        yhat_yzp[tmp_z] = float(torch.sum(loss[zp_index[tmp_z]])) / zp_len[tmp_z]
                        yhat_yzn[tmp_z] = float(torch.sum(loss1[zn_index[tmp_z]])) / zn_len[tmp_z]

                    # sorted_yhat_yz_p = sorted(yhat_yzp.items(), key=lambda item: item[1])
                    # sorted_yhat_yz_n = sorted(yhat_yzn.items(), key=lambda item: item[1])

                    # updated_prob = self.softmax_with_temperature(yhat_yzp, T=0.5)
                    #
                    # self.prob = [updated_prob[i] for i in range(len(self.prob))]
                    #06.20
                    yhat_yzp_items = list(yhat_yzp.values())
                    yhat_yzp_avg = sum(yhat_yzp_items) / len(yhat_yzp_items)
                    cur_grads = np.array([y - yhat_yzp_avg for y in yhat_yzp_items])
                    print('cur_grads', cur_grads)
                    self.prob = self.outer_optimizer.step(params=self.prob, grads=cur_grads)

                    self.prob[-1] += 1 - sum(self.prob)

                    print('prob', self.prob)

            for i, user in enumerate(users):
                start = time()
                posForUser = allPos[user]
                if len(posForUser) == 0:
                    continue
                for j in range(len(posForUser)):
                    posindex = j
                    positem = posForUser[posindex]

                    candidate_items = list(set(items) - set(posForUser))
                    negitem_category = np.random.choice(a=key_genre, size=1, replace=True, p=self.prob)[0]
                    candidate_list = list(Intersection(item_dict[negitem_category], candidate_items))
                    negitem = int(np.random.choice(candidate_list, size=1)[0])

                    S.append([user, positem, negitem])

            end = time()
            total = end - start
            print('sampling time', total)
            return np.array(S)

def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)

def Popularity_Sample_python(dataset, datapath):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
    np.array
    """
    dataset: BasicDataset
    users = [i for i in range(dataset.n_users)]
    items = [j for j in range(dataset.m_items)]

    allPos = dataset.allPos
    S = []
    pop_dict = pickle.load(open(datapath + '/' + 'item_idd_popularity_list.pkl', 'rb'))

    start = time()
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue

        candidate_items = list(set(items) - set(posForUser))
        items_popularity = [pow(pop_dict[i], 0.5) for i in candidate_items]
        total_popularity = sum(items_popularity)
        items_neg_prob = [i / total_popularity for i in items_popularity]

        for j in range(len(posForUser)):
            posindex = j
            positem = posForUser[posindex]

            negitem = int(np.random.choice(candidate_items, size=1, p=items_neg_prob)[0])
            S.append([user, positem, negitem])

    end = time()
    total = end - start
    print('sampling time', total)
    return np.array(S)


def fair_aware_prob(ss_group_prob, candidate_items, key_genre, item_idd_category):
    candidate_items_categroy = np.array([key_genre.index([j for j in item_idd_category[i] if j in key_genre][0]) \
                                         for i in candidate_items])

    prob = np.zeros(candidate_items_categroy.shape[0], dtype=np.float32)
    for k in range(len(key_genre)):
        key_genre_cnt = np.sum(candidate_items_categroy == k)
        prob[candidate_items_categroy == k] = float(ss_group_prob[k]) / float(key_genre_cnt)
    prob[-1] += float(1) - np.sum(prob)
    return prob


def UniformSample_python(datapath, dataset, key_genre, item_dict, neg_prob, neg_ratio):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
    np.array
    """
    dataset: BasicDataset
    users = [i for i in range(dataset.n_users)]
    items = [i for i in range(dataset.m_items)]
    item_idd_category = pickle.load(open(datapath + '/' + 'item_idd_genre_list.pkl', 'rb'))

    allPos = dataset.allPos
    S = []

    start = time()
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue

        candidate_items = list(set(items) - set(posForUser))
        # fair_aware_neg_prob = fair_aware_prob(neg_prob, candidate_items, key_genre, item_idd_category)
        for j in range(len(posForUser)):
            posindex = j
            positem = posForUser[posindex]

            # negitem_category = np.random.choice(a=key_genre, size=1, p=neg_prob)[0]
            # candidate_list = list(Intersection(item_dict[negitem_category], candidate_items))
            # negitem = int(np.random.choice(candidate_items, size=1, p=fair_aware_neg_prob)[0])
            negitem = int(np.random.choice(candidate_items, size=1)[0])

            S.append([user, positem, negitem])

    end = time()
    total = end - start
    print('sampling time', total)
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}-{world.config['method']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)


class CustomDataset(Dataset):
    """Custom Dataset.
    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (true labels) of data.
        z: A PyTorch tensor for z features (sensitive attributes) of data.
    """

    def __init__(self, x_tensor, pi_tensor, ni_tensor, z_tensor):
        """Initializes the dataset with torch tensors."""

        self.x = x_tensor
        self.pi = pi_tensor
        self.ni = ni_tensor
        self.z = z_tensor

    def __getitem__(self, index):
        """Returns the selected data based on the index information."""

        return (self.x[index], self.pi[index], self.ni[index],  self.z[index])

    def __len__(self):
        """Returns the length of data."""

        return len(self.x)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    # print('recall_n', recall_n)
    # recall = num_right_pre / num_truelike
    recall = np.sum(right_pred/recall_n)
    # precision = num_right_pre / topk_num
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def arp(sorted_items, k, datapath):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    f1 = pickle.load(open(datapath + '/' + 'item_idd_popularity_list.pkl', 'rb'))

    u_pop_list = []
    for u in range(sorted_items.shape[0]):
        l_u = sorted_items[u, :k]
        u_pop = 0
        u_len = len(l_u)
        for l in l_u:
            u_pop += f1[l]
        avg_pop = u_pop / u_len
        u_pop_list.append(avg_pop)

    return sum(u_pop_list)
    # return {'recall': recall, 'precision': precis}

def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    # ground_truth, right_predict, topk_num
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix

    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)

    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)

    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.

    return np.sum(ndcg)

def NDCG_at_k(predicted_list, ground_truth, k):
    dcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(predicted_list[:k])]
    dcg = np.sum(dcg_value)
    if len(ground_truth) < k:
        ground_truth += [0 for i in range(k - len(ground_truth))]
    idcg_value = [(v / log(i + 1 + 1, 2)) for i, v in enumerate(ground_truth[:k])]
    idcg = np.sum(idcg_value)
    return dcg / idcg

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        # print('groundTrue',  groundTrue)
        # print('predictTopK', predictTopK)
        # groundTrue[6866]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
