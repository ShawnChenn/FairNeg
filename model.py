"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
import pickle
from torch.autograd import Variable

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

    def bpr_loss_reg(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    def fair_reg_loss(self, users, pos, datapath, pklname, key_genre, pos_cpu):
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim, dtype=torch.float32)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim, dtype=torch.float32)
        nn.init.trunc_normal_(self.embedding_user.weight, mean=0, std=0.03)
        nn.init.trunc_normal_(self.embedding_item.weight, mean=0, std=0.03)
        # nn.init.normal_(self.embedding_user.weight, mean=0, std=0.03)
        # nn.init.normal_(self.embedding_item.weight, mean=0, std=0.03)
        # print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def  bpr_loss_reg(self, users, pos, neg, datapath, pklname, key_genre, pos_cpu):
        users_emb = self.embedding_user(users.long())

        pos_emb   = self.embedding_item(pos.long())

        neg_emb   = self.embedding_item(neg.long())

        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2))/float(len(users))

        zp_attribute = []
        f1 = pickle.load(open(datapath + '/' + pklname, 'rb'))

        for p in np.array(pos_cpu):
            gs = f1[p]
            for g in gs:
                if g in key_genre:
                    zp_attribute.append(key_genre.index(g))
                    break

        z_item = list(set(zp_attribute))
        zp_index = {}
        zp_mask = {}
        yhat_yzp = {}
        zp_len = {}

        for tmp_z in z_item:
            zp_mask[tmp_z] = torch.Tensor([1 if i == tmp_z else 0 for i in zp_attribute])

        for tmp_z in z_item:
            zp_index[tmp_z] = (zp_mask[tmp_z] == 1).nonzero().squeeze()

        for tmp_z in z_item:
            zp_len[tmp_z] = len(zp_index[tmp_z])

        for tmp_z in z_item:
            yhat_yzp[tmp_z] = torch.sum(pos_scores[zp_index[tmp_z]]) / zp_len[tmp_z]

        if len(z_item) == 2:
            fair_reg_loss = torch.square(yhat_yzp[z_item[0]] - yhat_yzp[z_item[1]])
        else:
            fair_reg_loss = torch.square(yhat_yzp[z_item[0]] - yhat_yzp[z_item[0]])
        # fair_reg_loss = 0

        return loss, reg_loss, fair_reg_loss

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())

        pos_emb = self.embedding_item(pos.long())

        neg_emb = self.embedding_item(neg.long())

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))

        # eo_loss = self.get_pos_logit(users.cuda(), pos.cuda())
        # zp_attribute = []
        # f1 = pickle.load(open(datapath + '/' + pklname, 'rb'))
        # for p in np.array(pos_cpu):
        #     gs = f1[p]
        #     for g in gs:
        #         if g in key_genre:
        #             zp_attribute.append(key_genre.index(g))
        #             break
        #
        # z_item = list(set(zp_attribute))
        # zp_index = {}
        # zp_mask = {}
        # yhat_yzp = {}
        # zp_len = {}
        #
        # for tmp_z in z_item:
        #     zp_mask[tmp_z] = torch.Tensor([1 if i == tmp_z else 0 for i in zp_attribute])
        #
        # for tmp_z in z_item:
        #     zp_index[tmp_z] = (zp_mask[tmp_z] == 1).nonzero().squeeze()
        #
        # for tmp_z in z_item:
        #     zp_len[tmp_z] = len(zp_index[tmp_z])
        #
        # for tmp_z in z_item:
        #     yhat_yzp[tmp_z] = torch.sum(eo_loss[zp_index[tmp_z]]) / zp_len[tmp_z]
        #
        # yhat_yzp_avg = torch.mean(eo_loss)
        #
        # loss1 = nn.L1Loss()
        # fair_reg_loss = 0
        # for z in z_item:
        #     fair_reg_loss += loss1(yhat_yzp[z], yhat_yzp_avg)
        # return loss, reg_loss, fair_reg_loss
        return loss, reg_loss, 0


    def getlogit(self, users, pos, neg):
        users = users.long()
        pos = pos.long()
        neg = neg.long()

        users_emb = self.embedding_user(users)
        pos_emb = self.embedding_item(pos)
        neg_emb = self.embedding_item(neg)

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        target1 = torch.ones((pos_scores.shape[0])).cuda()
        loss1 = F.binary_cross_entropy(F.sigmoid(pos_scores), target1, reduce=False)

        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        target2 = torch.zeros((neg_scores.shape[0])).cuda()
        loss2 = F.binary_cross_entropy(F.sigmoid(neg_scores), target2, reduce=False)
        return loss1, loss2

    def get_pos_logit(self, users, pos):
        users = users.long()
        pos = pos.long()

        users_emb = self.embedding_user(users)
        pos_emb = self.embedding_item(pos)

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        target1 = torch.ones((pos_scores.shape[0])).cuda()
        loss1 = F.binary_cross_entropy(F.sigmoid(pos_scores), target1, reduce=False)
        return loss1

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # bpr loss
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # bce loss
        # target1 = torch.ones((pos_scores.shape[0])).cuda()
        # loss1 = F.binary_cross_entropy(F.sigmoid(pos_scores), target1, reduce=True)
        # target2 = torch.zeros((neg_scores.shape[0])).cuda()
        # loss2 = F.binary_cross_entropy(F.sigmoid(neg_scores), target2, reduce=True)
        # loss = loss1 + loss2

        # cce loss
        # pos_loss = torch.relu(1 - pos_scores)
        # neg_loss = torch.relu(neg_scores - 0.7)
        # loss = torch.mean(pos_loss) + 0.3 * torch.mean(neg_loss)
        return loss, reg_loss

    def getlogit(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        target1 = torch.ones((pos_scores.shape[0])).cuda()
        loss1 = F.binary_cross_entropy(F.sigmoid(pos_scores), target1, reduce=False)

        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        target2 = torch.zeros((neg_scores.shape[0])).cuda()
        loss2 = F.binary_cross_entropy(F.sigmoid(neg_scores), target2, reduce=False)
        return loss1, loss2

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
