import random

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn import preprocessing
import pdb
import torch
import h5py
import scipy.io as sio

torch.multiprocessing.set_sharing_strategy('file_system')

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]

        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count


def get_loader_flickr(batch_size):
    path = './datasets/MIRFlickr/'

    # x: images   y:tags   L:labels
    train_set = sio.loadmat(path + 'mir_train.mat')
    train_L = np.array(train_set['L_tr'], dtype=np.float)
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)

    test_set = sio.loadmat(path + 'mir_query.mat')
    query_L = np.array(test_set['L_te'], dtype=np.float)
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)

    db_set = sio.loadmat(path + 'mir_database.mat')
    retrieval_L = np.array(db_set['L_db'], dtype=np.float)
    retrieval_x = np.array(db_set['I_db'], dtype=np.float)
    retrieval_y = np.array(db_set['T_db'], dtype=np.float)


    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in ['query', 'train', 'database']}

    return dataloader, (train_x, train_y, train_L)


def get_loader_nus(batch_size):
    path = './datasets/NUS-WIDE/'

    # x: images   y:tags   L:labels
    train_set = sio.loadmat(path + 'nus_train.mat')
    train_L = np.array(train_set['L_tr'], dtype=np.float)
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)

    test_set = sio.loadmat(path + 'nus_query.mat')
    query_L = np.array(test_set['L_te'], dtype=np.float)
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)

    db_set = sio.loadmat(path + 'nus_database.mat')
    retrieval_L = np.array(db_set['L_db'], dtype=np.float)
    retrieval_x = np.array(db_set['I_db'], dtype=np.float)
    retrieval_y = np.array(db_set['T_db'], dtype=np.float)

    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in ['query', 'train', 'database']}

    return dataloader, (train_x, train_y, train_L)


def get_loader_coco(batch_size):
    path = './datasets/MSCOCO/'

    # x: images   y:tags   L:labels
    train_set = sio.loadmat(path + 'COCO_train.mat')
    train_L = np.array(train_set['L_tr'], dtype=np.float)
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)

    test_set = sio.loadmat(path + 'COCO_query.mat')
    query_L = np.array(test_set['L_te'], dtype=np.float)
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)

    db_set = h5py.File(path + 'COCO_database.mat', 'r', libver='latest', swmr=True)
    retrieval_L = np.array(db_set['L_db'], dtype=np.float).T
    retrieval_x = np.array(db_set['I_db'], dtype=np.float).T
    retrieval_y = np.array(db_set['T_db'], dtype=np.float).T
    db_set.close()

    imgs = {'train': train_x, 'query': query_x, 'database': retrieval_x}
    texts = {'train': train_y, 'query': query_y, 'database': retrieval_y}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database']}

    shuffle = {'query': False, 'train': True, 'database': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in
                  ['query', 'train', 'database']}

    return dataloader, (train_x, train_y, train_L)
