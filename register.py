import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'yelp_multiclass':
    dataset = dataloader.myLoader(path="/mnt/data/recommend_dataset/yelp_pkl")
elif world.dataset == 'ml1m-2':
    dataset = dataloader.myLoader(path="/mnt/data/recommend_dataset/ICDE2023_data_preprocessing/ml1m-2")
elif world.dataset == 'ml1m-4':
    dataset = dataloader.myLoader(path="/mnt/data/recommend_dataset/ICDE2023_data_preprocessing/ml1m-4")
elif world.dataset == 'amazon':
    dataset = dataloader.myLoader(path="/mnt/data/recommend_dataset/amazon")
elif world.dataset == 'amazon-2':
    dataset = dataloader.myLoader(path="/mnt/data/recommend_dataset/ICDE2023_data_preprocessing/amazon-2")
elif world.dataset == 'amazon-4':
    dataset = dataloader.myLoader(path="/mnt/data/recommend_dataset/ICDE2023_data_preprocessing/amazon-4")
elif world.dataset == 'yelp-2':
    dataset = dataloader.myLoader(path="/mnt/data/recommend_dataset/ICDE2023_data_preprocessing/yelp-2")
elif world.dataset == 'yelp-4':
    dataset = dataloader.myLoader(path="/mnt/data/recommend_dataset/ICDE2023_data_preprocessing/yelp-4")
elif world.dataset == 'amazon_book':
    dataset = dataloader.myLoader(path="/mnt/data/recommend_dataset")

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}