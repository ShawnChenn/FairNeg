import sys
sys.path.append('../')
import world
import utils
from world import cprint
import torch
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import pickle
import register
from utils import timer
from register import dataset
from torch.utils.data import Dataset, DataLoader, Sampler
from fairsampler import FairBatch
from Procedure import stats_item
from parse import parse_args

args = parse_args()

datapath = args.data_path
key_genre = args.key_genre
balance_ratio = args.br

pklname = 'item_idd_genre_list.pkl'
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.method + '-' + world.model_name + "-" + world.dataset + '-' + str(balance_ratio)))
else:
    w = None
    world.cprint("not enable tensorflowboard")

item_dict = {}
item_genre = pickle.load(open(datapath + '/' + pklname, 'rb'))
for i in range(len(item_genre)):
    genres = item_genre[i]
    intersec_list = [j for j in genres if j in key_genre]
    if len(intersec_list) == 1:
        key = intersec_list[0]
        if key in item_dict.keys():
            item_dict[key].append(i)
        else:
            item_dict[key] = [i]

key_num = [len(item_dict[k]) for k in key_genre]
neg_prob = []
for i in range(len(key_genre) - 1):
    neg_prob.append(key_num[i] / sum(key_num))
neg_prob.append(1 - sum(neg_prob))

print('neg prob', neg_prob)

pres = None
prob = None
try:
    Sampler = utils.hardNeg_Sample_python(dataset, datapath, pklname, Recmodel, neg_prob, key_genre)
    best_valid_score = 0.0
    cur_step = 0
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0 and epoch != 0:
            cprint("[valid]")
            results0, results1 = Procedure.Valid(datapath, key_genre, dataset, Recmodel, epoch, w, world.config['multicore'])
            valid_score = results0['F1'][-1]

            cprint("[test]")
            results = Procedure.Test(datapath, key_genre, dataset, Recmodel, epoch, w, world.config['multicore'])

            from utils import early_stopping
            best_valid_score, cur_step, stop_flag, update_flag = early_stopping(
                valid_score,
                best_valid_score,
                cur_step,
                max_step=10,
            )
            print('current best valid score', best_valid_score)

            if update_flag:
                print('saving best model')
                torch.save(Recmodel.state_dict(), weight_file)

            if stop_flag:
                stop_output = 'Finished training, best eval result in epoch %d' % \
                              (epoch - cur_step * 10)
                print(stop_output)
                break

        # train process
        S = Sampler.sample_neg(dataset, datapath, key_genre, pklname, Recmodel, balance_ratio)
        # S = utils.UniformSample_original(dataset, key_genre, item_dict, neg_prob, neg_ratio=1)
        if epoch % 10 == 0:
            stats_item(datapath, key_genre, S[:, 1], S[:, 2], pklname)

        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()

        item_attribute_z1 = []
        item_attribute_z2 = []

        f1 = pickle.load(open(datapath + '/' + pklname, 'rb'))
        for p in S[:, 1]:
            gs = f1[p]
            for g in gs:
                if g in key_genre:
                    item_attribute_z1.append(key_genre.index(g))

        for p in S[:, 2]:
            gs = f1[p]
            for g in gs:
                if g in key_genre:
                    item_attribute_z2.append(key_genre.index(g))

        Items_z1 = torch.Tensor(item_attribute_z1).long()
        Items_z2 = torch.Tensor(item_attribute_z2).long()

        train_data = utils.CustomDataset(users, posItems, negItems, Items_z1)
        # 0620
        # sampler1 = FairBatch(Recmodel, users, posItems, negItems, Items_z1, Items_z2,  world.config['bpr_batch_size'], 0.01, 'eqopp', prob=prob)

        train_loader = DataLoader(train_data, batch_size=world.config['bpr_batch_size'], num_workers=0)

        # train_loader = DataLoader(train_data, batch_sampler=sampler1, num_workers=0)
        total_batch = len(users) // world.config['bpr_batch_size'] + 1
        aver_loss = 0.
        Recmodel.train()
        for batch_i, (batch_users, batch_pos, batch_neg, _) in enumerate(train_loader):
            cri = bpr.stageOne(batch_users.to(world.device), batch_pos.to(world.device), batch_neg.to(world.device), datapath, pklname, key_genre, batch_pos)
            aver_loss += cri
            if world.tensorboard:
                w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

        # prob = sampler1.lb
        # print('epoch', epoch, prob)
        aver_loss = aver_loss / total_batch
        time_info = timer.dict()
        timer.zero()
        output_information = f"loss{aver_loss:.3f}-{time_info}"
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
finally:
    if world.tensorboard:
        w.close()