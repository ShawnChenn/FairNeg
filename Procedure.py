'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import pickle
import operator

CORES = multiprocessing.cpu_count() // 2

def stats_item(path, key_genre, pitems, nitems, pklname):
    f1 = pickle.load(open(path + '/' + pklname, 'rb'))
    pitem_dist = dict()
    nitem_dist = dict()

    for p in pitems:
        gs = f1[p]
        # gs = f1[p].split(',')
        for g in gs:
            # g = g.strip(' ')
            if g in key_genre:
                if g in pitem_dist.keys():
                    pitem_dist[g] += 1
                else:
                    pitem_dist[g] = 1

    for n in nitems:
        gs = f1[n]
        # gs = f1[n].split(',')
        for g in gs:
            # g = g.strip(' ')
            if g in key_genre:
                if g in nitem_dist.keys():
                    nitem_dist[g] += 1
                else:
                    nitem_dist[g] = 1

    print('positive items distribution', sorted(pitem_dist.items(), key=operator.itemgetter(1), reverse=True))
    print('===============' + '\n')
    print('negative items distribution', sorted(nitem_dist.items(), key=operator.itemgetter(1), reverse=True))

def ranking_analysis(path, key_genre, rating_list, groundTrue_list, k_num):
    """
    rating_list
    groundTruelist
    """
    f1 = pickle.load(open(path + '/item_idd_genre_list.pkl', 'rb'))
    # each genre item count in gt
    count_dict = dict()
    # each genre right predicted items count as for prediction result
    right_predicted_dict = dict()
    # each genre predicted items count as for prediction result
    predicted_dict = dict()

    precision = dict()
    recall = dict()

    for X in zip(rating_list, groundTrue_list):
        predict_items = X[0]
        groundTrue = X[1]
        pred_onehot = []

        for i in range(len(groundTrue)):
            gt = groundTrue[i]
            predictTopK = predict_items[i]
            po = list(map(lambda x: x in gt, predictTopK))
            pon = [a if b == 1.0 else -1 for a, b in zip(predictTopK, po)]
            pred_onehot.append(pon)

        for gt in groundTrue:
            for id in gt:
                genres = f1[id]
                # print(genre)
                # for gr in genres.split(','):
                for gr in genres:
                    k = gr
                    # k = gr.strip(' ')
                    if k in key_genre:
                        if k in count_dict.keys():
                            count_dict[k] += 1
                        else:
                            count_dict[k] = 1

        for pi in predict_items:
            for id in pi[:k_num]:
                genres = f1[id.item()]
                for gr in genres:
                # for gr in genres.split(','):
                    k = gr
                    if k in key_genre:
                        if k in predicted_dict.keys():
                            predicted_dict[k] += 1
                        else:
                            predicted_dict[k] = 1

        for pi in pred_onehot:
            for id in pi[:k_num]:
                if id >= 0:
                    genres = f1[id.item()]
                    for gr in genres:
                    # for gr in genres.split(','):
                        k = gr
                    #     k = gr.strip(' ')
                        if k in key_genre:
                            if k in right_predicted_dict.keys():
                                right_predicted_dict[k] += 1
                            else:
                                right_predicted_dict[k] = 1

    for k in count_dict.keys():
        if k not in right_predicted_dict.keys():
            rpre = 0
        else:
            rpre = right_predicted_dict[k]

        if k not in predicted_dict.keys():
            precision[k] = 0
        else:
            precision[k] = rpre / predicted_dict[k]

        recall[k] = rpre / count_dict[k]
    print(k_num, precision, recall)

    tmp = list(recall.values())
    rstd = np.std(tmp) / (np.mean(tmp) + 1e-10)

    return {'recall': recall,
            'precision': precision,
            'recall_rstd': rstd}

def test_one_batch(X, datapath):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    # print('sorted_items, groundTrue', sorted_items.shape, len(groundTrue))
    # (100, 20) 100
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, arps = [], [], [], []

    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        # ndcg.append(utils.NDCG_at_k(sorted_items, groundTrue, k))
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))

        # arp = utils.arp(sorted_items, k, datapath)
        arp = 0
        arps.append(arp)

    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg),
            'arp': np.array(arps)}
            
def Test(datapath, key_genre, dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    user_results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'F1': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'arp': np.zeros(len(world.topks))}

    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        u_batch_size = len(users)
        if len(users) % u_batch_size == 0:
            total_batch = len(users) // u_batch_size
        else:
            total_batch = len(users) // u_batch_size + 1

        print('u_batch_size', u_batch_size)
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        #print('len(rating_list), len(rating_list[0])', len(rating_list), len(rating_list[0]))
        #total_batch_num, batch_user_size
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            print('using multicore...', multicore)
            pre_results = []
            for x in X:
                # EACH BATCH CALCULATES RECALL, PRECISION, NDCG
                pre_results.append(test_one_batch(x, datapath))

            pre1_results = []
            for k in world.topks:
                pre1_results.append(ranking_analysis(datapath, key_genre, rating_list, groundTrue_list, k))

        scale = float(u_batch_size/len(users))
        for result in pre_results:
            user_results['recall'] += result['recall']
            user_results['precision'] += result['precision']
            user_results['ndcg'] += result['ndcg']
            user_results['arp'] += result['arp']

        user_results['recall'] /= float(len(users))
        user_results['precision'] /= float(len(users))
        user_results['F1'] = [2 * user_results['recall'][i] * user_results['precision'][i] / (user_results['recall'][i] + user_results['precision'][i]) for i in range(len(world.topks))]
        user_results['ndcg'] /= float(len(users))
        user_results['arp'] /= float(len(users))

        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): user_results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): user_results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/F1@{world.topks}',
                          {str(world.topks[i]): user_results['F1'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): user_results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/ARP@{world.topks}',
                          {str(world.topks[i]): user_results['arp'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/item_recall@{world.topks}',
                          {str(world.topks[i]): pre1_results[i]['recall_rstd'] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        return user_results, pre1_results

def Valid(datapath, key_genre, dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    validDict: dict = dataset.validDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'F1': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'arp': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(validDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        u_batch_size = len(users)

        if len(users) % u_batch_size == 0:
            total_batch = len(users) // u_batch_size
        else:
            total_batch = len(users) // u_batch_size + 1
        # print('total_batch', total_batch)

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [validDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        # print(total_batch, len(users_list))
        assert total_batch == len(users_list)

        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            print('not using multicore...')
            pre_results = []
            for x in X:
                # EACH BATCH CALCULATES RECALL, PRECISION, NDCG
                pre_results.append(test_one_batch(x, datapath))

            pre1_results = []
            for k in world.topks:
                pre1_results.append(ranking_analysis(datapath, key_genre, rating_list, groundTrue_list, k))

        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['arp'] += result['arp']

        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['F1'] = [2 * results['recall'][i] * results['precision'][i] / (results['recall'][i] + results['precision'][i]) for i in range(len(world.topks))]

        results['ndcg'] /= float(len(users))
        results['arp'] /= float(len(users))

        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Valid/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Valid/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'valid/F1@{world.topks}',
                          {str(world.topks[i]): results['F1'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Valid/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Valid/ARP@{world.topks}',
                          {str(world.topks[i]): results['arp'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'valid/item_recall@{world.topks}',
                          {str(world.topks[i]): pre1_results[i]['recall_rstd'] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print('recommendation performance')
        print(results)
        print('fairness performance')
        print(pre1_results)
        return results, pre1_results