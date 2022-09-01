import os
import logging
import torch
from torch.nn import DataParallel
from tqdm import tqdm
from Earlystopping import EarlyStopping

class yAwareCLModel:

    def __init__(self, net, loss, loader_train, loader_val, loader_test, config, n_iter, scheduler=None):
        """
        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        loader_train, loader_val: pytorch DataLoaders for training/validation
        config: Config object with hyperparameters
        scheduler (optional)
        """
        super().__init__()
        self.logger = logging.getLogger("yAwareCL")
        self.model = net
        self.loss = loss
        self.config = config

        if config.layer_control == 'tune_all':
            self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        elif config.layer_control == 'freeze':
            for name, param in net.named_parameters():
                if param.requires_grad and 'classifier' not in name:
                    param.requires_grad = False
            self.optimizer = torch.optim.Adam(net.classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        else: # layer_control == 'tune_diff':
            if config.model == 'DenseNet':
                self.optimizer = torch.optim.Adam([
                    {"params": net.features.conv0.parameters(), "lr": config.lr*1e-3},
                    {"params": net.features.denseblock1.parameters(), "lr": config.lr*1e-3},
                    {"params": net.features.transition1.parameters(), "lr": config.lr*1e-3},

                    {"params": net.features.denseblock2.parameters(), "lr": config.lr*1e-2},
                    {"params": net.features.transition2.parameters(), "lr": config.lr*1e-2},

                    {"params": net.features.denseblock3.parameters(), "lr": config.lr*1e-1},
                    {"params": net.features.transition3.parameters(), "lr": config.lr*1e-1},

                    {"params": net.features.denseblock4.parameters(), "lr": config.lr},
                    {"params": net.classifier.parameters(), "lr": config.lr},
                    ], lr=config.lr, weight_decay=config.weight_decay)
            else: # config.model == 'UNet':
                self.optimizer = torch.optim.Adam([
                    {"params": net.up.parameters(), "lr": config.lr*1e-2},
                    {"params": net.down.parameters(), "lr": config.lr*1e-1},
                    {"params": net.classifier.parameters(), "lr": config.lr},
                    ], lr=config.lr, weight_decay=config.weight_decay)
        
        self.loader = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        
        self.device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")

        self.task_name = config.task_name
        self.train_num = config.train_num
        self.layer_control = config.layer_control
        self.pretrained_path = config.pretrained_path
        
        self.n_iter = n_iter
        self.scheduler = scheduler

        self.model = DataParallel(self.model).to(self.device)
        if config.pretrained_path != 'None':
            self.load_model(config.pretrained_path)



    def fine_tuning(self):
        print(self.loss)
        print(self.optimizer)
        if self.n_iter is not None:
            n_iter = 'CV' + str(self.n_iter)
        else:
            n_iter = 'model'
        
        path1 = './ckpts/{0}'.format(self.task_name.replace('/', ''))
        isExist = os.path.exists(path1)
        if not isExist:
            os.makedirs(path1)
        path2 = './ckpts/{0}/{1}'.format(self.task_name.replace('/', ''), 
                                         str(self.pretrained_path).split('/')[-1].split('.')[0])
        isExist = os.path.exists(path2)
        if not isExist:
            os.makedirs(path2)

        early_stopping = EarlyStopping(patience = self.config.patience, 
                                       path = './ckpts/{0}/{1}/{1}_{2}_{3}_{4}.pt'.format(self.task_name.replace('/', ''), 
                                                                                          str(self.pretrained_path).split('/')[-1].split('.')[0], 
                                                                                          self.layer_control[0],
                                                                                          self.train_num,
                                                                                          n_iter))
        for epoch in range(self.config.nb_epochs):
            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss, training_acc = 0, 0
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels) in self.loader:
                pbar.update()
                labels = torch.flatten(labels).type(torch.LongTensor)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(inputs)
                if self.config.task_type == 'reg':
                    labels = labels.to(torch.float32)
                batch_loss = self.loss(y, labels)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += batch_loss.item()*inputs.size(0)
                if self.config.task_type == 'cls':
                    _, predicted = torch.max(y, 1)
                    training_acc += (predicted == labels).sum().item()
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss, val_acc = 0, 0
            self.model.eval()
            with torch.no_grad():
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    labels = torch.flatten(labels).type(torch.LongTensor)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    y = self.model(inputs)
                    if self.config.task_type == 'reg':
                        labels = labels.to(torch.float32)
                    batch_loss = self.loss(y, labels)
                    val_loss += batch_loss.item()*inputs.size(0)
                    if self.config.task_type == 'cls':
                        _, predicted = torch.max(y, 1)
                        val_acc += (predicted == labels).sum().item()
            pbar.close()

            print("\nEpoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                  epoch+1, self.config.nb_epochs, training_loss / len(self.loader.dataset), val_loss / len(self.loader_val.dataset)), flush=True)
            if self.config.task_type == 'cls':
                print("Training accuracy: {:.2f}%\t Validation accuracy: {:.2f}%\t".format(
                      100 * training_acc / len(self.loader.dataset), 100 * val_acc / len(self.loader_val.dataset)), flush=True)
            
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("[ Early stopped ]")
                self.epoch_data = epoch + 1
                break
            else:
                self.epoch_data = epoch + 1

            if self.scheduler is not None:
                self.scheduler.step()

        model_info = self.model.load_state_dict(torch.load('./ckpts/{0}/{1}/{1}_{2}_{3}_{4}.pt'.format(self.task_name.replace('/', ''), 
                                                                                                       str(self.pretrained_path).split('/')[-1].split('.')[0], 
                                                                                                       self.layer_control[0],
                                                                                                       self.train_num,
                                                                                                       n_iter)))
        print('self.model loading info: {}'.format(model_info))

        ## Test step
        nb_batch = len(self.loader_test)
        pbar = tqdm(total=nb_batch, desc="Test")
        test_loss, test_acc = 0, 0
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        self.model.eval()
        with torch.no_grad():
            for (inputs, labels) in self.loader_test:
                pbar.update()
                labels = torch.flatten(labels).type(torch.LongTensor)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                y = self.model(inputs)

                if self.config.task_type == 'reg':
                    labels = labels.to(torch.float32)
                    outGT = torch.cat((outGT, labels), 0)
                    outPRED = torch.cat((outPRED, y), 0)

                if self.config.task_type == 'cls':
                    m = torch.nn.Softmax(dim=1)
                    output = m(y)
                    if int(labels) == 0:
                        onehot = torch.LongTensor([[1, 0]])
                    elif int(labels) == 1:
                        onehot = torch.LongTensor([[0, 1]])
                    onehot = onehot.cuda()
                    outGT = torch.cat((outGT, onehot), 0)
                    outPRED = torch.cat((outPRED, output), 0)
                    _, predicted = torch.max(y, 1)
                    test_acc += (predicted == labels).sum().item()

                batch_loss = self.loss(y, labels)
                test_loss += batch_loss.item()*inputs.size(0)
                    
        pbar.close()
        if self.config.task_type == 'cls':
            print("\n\nTest loss: {:.4f}\t Test accuracy: {:.2f}%\t".format(
                  test_loss / len(self.loader_test.dataset), 100 * test_acc / len(self.loader_test.dataset)), flush=True)
        else:
            print("\n\nTest loss: {:.4f}".format(test_loss / len(self.loader_test.dataset)), flush=True)

        return outGT, outPRED, test_acc / len(self.loader_test.dataset), self.epoch_data



    def load_model(self, path):
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except BaseException as e:
            print('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected = self.model.load_state_dict(checkpoint.state_dict())
                    print('Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
                        print('Model loading info: {}'.format(unexpected))
                else:
                    unexpected = self.model.load_state_dict(checkpoint)
                    print('Model loading info: {}'.format(unexpected))
            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))
