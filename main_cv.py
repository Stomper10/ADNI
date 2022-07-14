import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
import time
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from config import Config
from dataset import ADNI_Dataset
from models.unet import UNet
from models.densenet import densenet121
from yAwareContrastiveLearning import yAwareCLModel
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    now = datetime.datetime.now()
    print("[main_cv.py started at {0}]".format(now.strftime('%Y-%m-%d %H:%M:%S')))
    start_time = time.time()
    parser = argparse.ArgumentParser() 
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Set the pretrained model path.")   
    parser.add_argument("--train_num", type=int, required=True,
                        help="Set the number of training samples.")                        
    parser.add_argument("--task_name", type=str, required=True,
                        help="Set the name of the fine-tuning task. (e.g. AD/MCI)")
    parser.add_argument("--layer_control", type=str, choices=['tune_all', 'freeze', 'tune_diff'], required=True,
                        help="Set pretrained weight layer control option.")
    parser.add_argument("--random_seed", type=int, required=False, default=0,
                        help="Random seed for reproduction.")
    args = parser.parse_args()
    config = Config(args)

    train_num = config.train_num
    task_name = config.task_name
    layer_control = config.layer_control
    print('Pretrained path:', config.pretrained_path)

    # Control randomness for reproduction
    if config.random_seed != None:
        random_seed = config.random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        #os.environ["PYTHONHASHSEED"] = str(random_seed)
        #torch.manual_seed(random_seed)
        #torch.cuda.manual_seed(random_seed)
        #torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

    labels = pd.read_csv(config.label)
    label_name = config.label_name # 'Dx.new'
    print('Task: Fine-tuning for {0}'.format(task_name))
    print('Task type: {0}'.format(config.task_type))
    print('N = {0}'.format(train_num))

    if config.task_type == 'cls':
        task_include = task_name.split('/')
        assert len(task_include) == 2, 'Set two labels.'
        assert config.num_classes == 2, 'Set config.num_classes == 2'

        data_1 = labels[labels[label_name] == task_include[0]]
        data_2 = labels[labels[label_name] == task_include[1]]
        limit = len(data_1) if len(data_1) <= len(data_2) else len(data_2)
        data_1 = data_1.sample(frac=1, random_state=random_seed)[:limit]
        data_2 = data_2.sample(frac=1, random_state=random_seed)[:limit]

        len_1_train = round(train_num*0.5)
        len_2_train = train_num - len_1_train
        len_1_valid = round(int(train_num*config.valid_ratio)*0.5)
        len_2_valid = int(train_num*config.valid_ratio) - len_1_valid
        assert train_num*(1+config.valid_ratio) < limit*2, 'Not enough data to make balanced set.'

        train1, valid1, test1 = np.split(data_1, [len_1_train, len_1_train + len_1_valid])
        train2, valid2, test2 = np.split(data_2, [len_2_train, len_2_train + len_2_valid])
        label_train = pd.concat([train1, train2]).sample(frac=1, random_state=random_seed)
        label_valid = pd.concat([valid1, valid2]).sample(frac=1, random_state=random_seed)
        label_test = pd.concat([test1, test2]).sample(frac=1, random_state=random_seed)
        assert len(label_test) >= 100, 'Not enough test data. (Total: {0})'.format(len(label_test))

        print('\nTrain data info:\n{0}\nTotal: {1}\n'.format(label_train[label_name].value_counts().sort_index(), len(label_train)))
        print('Valid data info:\n{0}\nTotal: {1}\n'.format(label_valid[label_name].value_counts().sort_index(), len(label_valid)))
        print('Test data info:\n{0}\nTotal: {1}\n'.format(label_test[label_name].value_counts().sort_index(), len(label_test)))

        label_train.to_csv('./csv/ADNI_{0}{1}train.csv'.format(task_name.replace('/', ''), str(train_num)[0]), index=False)
        label_valid.to_csv('./csv/ADNI_{0}{1}valid.csv'.format(task_name.replace('/', ''), str(train_num)[0]), index=False)
        label_test.to_csv('./csv/ADNI_{0}{1}test.csv'.format(task_name.replace('/', ''), str(train_num)[0]), index=False)

        label_train[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
        label_valid[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
        label_test[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)

    else: # config.task_type = 'reg'
        task_include = task_name.split('/')
        assert len(task_include) == 1, 'Set only one label.'
        assert config.num_classes == 1, 'Set config.num_classes == 1'

        labels = pd.read_csv(config.label)
        labels = labels[(np.abs(stats.zscore(labels[label_name])) < 3)] # remove outliers w.r.t. z-score > 3
        assert train_num*(1+config.valid_ratio) <= len(labels), 'Not enough valid data. Set smaller --train_num or smaller config.valid_ratio in config.py.'

        label_train, label_valid, label_test = np.split(labels.sample(frac=1, random_state=random_seed), 
                                                        [train_num, int(train_num*(1+config.valid_ratio))])
        
        print('\nTrain data info:\nTotal: {0}\n'.format(len(label_train)))
        print('Valid data info:\nTotal: {0}\n'.format(len(label_valid)))
        print('Test data info:\nTotal: {0}\n'.format(len(label_test)))

    # Final evaluation
    SPLITS = config.splits
    n_iter = 0
    skf = StratifiedKFold(n_splits = SPLITS)

    label_tv = pd.concat([label_train, label_valid])
    aurocMean_list, acc_list, epoch_list = [], [], []
    mse_list = []
    for train_idx, valid_idx in skf.split(label_tv, label_tv[label_name]):
        n_iter += 1
        print('\n<<< StratifiedKFold: {0}/{1} >>>'.format(n_iter, SPLITS))
        label_train = label_tv.iloc[train_idx]
        label_valid = label_tv.iloc[valid_idx]

        dataset_train = ADNI_Dataset(config, label_train, data_type='train')
        dataset_val = ADNI_Dataset(config, label_valid, data_type='valid')
        dataset_test = ADNI_Dataset(config, label_test, data_type='test')

        loader_train = DataLoader(dataset_train,
                                  batch_size=config.batch_size,
                                  sampler=RandomSampler(dataset_train),
                                  collate_fn=dataset_train.collate_fn,
                                  pin_memory=config.pin_mem,
                                  num_workers=config.num_cpu_workers)
        loader_val = DataLoader(dataset_val,
                                batch_size=config.batch_size,
                                sampler=RandomSampler(dataset_val),
                                collate_fn=dataset_val.collate_fn,
                                pin_memory=config.pin_mem,
                                num_workers=config.num_cpu_workers)
        loader_test = DataLoader(dataset_test,
                                 batch_size=1,
                                 sampler=RandomSampler(dataset_test),
                                 collate_fn=dataset_test.collate_fn,
                                 pin_memory=config.pin_mem,
                                 num_workers=config.num_cpu_workers)

        if config.model == "DenseNet":
            net = densenet121(mode="classifier", drop_rate=0.0, num_classes=config.num_classes)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="classif")
        else:
            raise ValueError("Unkown model: %s"%config.model)

        if config.task_type == 'cls':
            loss = CrossEntropyLoss()
        else: # config.task_type == 'reg':
            loss = MSELoss()

        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, n_iter)
        outGT, outPRED, acc, epoch_data = model.fine_tuning()

        if config.task_type == 'cls':
            outGTnp = outGT.cpu().numpy()
            outPREDnp = outPRED.cpu().numpy()
            print('<< Test Results: AUROC >>')
            outAUROC = []
            for i in range(config.num_classes):
                outAUROC.append(roc_auc_score(outGTnp[:, i], outPREDnp[:, i]))
            aurocMean = np.array(outAUROC).mean()
            print('MEAN', ': {:.4f}\n'.format(aurocMean))
            aurocMean_list.append(aurocMean)
            acc_list.append(acc)
        else: # config.task_type == 'reg':
            outGTnp = outGT.cpu().numpy()
            outPREDnp = outPRED.cpu().numpy()
            mse_list.append(mean_squared_error(outGTnp, outPREDnp))
            print('<< Test Results >>')
            print('MSE: {:.2f}'.format(mean_squared_error(outGTnp, outPREDnp)))
            print('MAE: {:.2f}'.format(mean_absolute_error(outGTnp, outPREDnp)))
            print('RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(outGTnp, outPREDnp))))
            print('R2-score: {:.4f}'.format(r2_score(outGTnp, outPREDnp)))

        epoch_list.append(epoch_data)

    if config.task_type == 'cls':
        aurocMean_arr = np.array(aurocMean_list)
        acc_list_arr = np.array(acc_list)
        epoch_list_arr = np.array(epoch_list)
        print('<<< Stratified {0} Fold Test mean AUC: {1:.4f}, std: {2:.4f} \
               / Test mean ACC: {3:.4f}, std: {4:.4f} >>>'.format(SPLITS, np.mean(aurocMean_arr), np.std(aurocMean_arr), np.mean(acc_list_arr), np.std(acc_list_arr)))
    else: # config.task_type == 'reg':
        print('<<< Stratified {0} Fold Test mean MSE: {1:.4f}, std: {2:.4f}'.format(SPLITS, np.mean(mse_list), np.std(mse_list)))

    print('Early stopped epoch info: {0}, mean : {1}'.format(epoch_list, np.mean(epoch_list_arr)))
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')
    now = datetime.datetime.now()
    print("[main_cv.py finished at {0}]\n".format(now.strftime('%Y-%m-%d %H:%M:%S')))
