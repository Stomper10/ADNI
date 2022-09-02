class Config:

    def __init__(self, args):
        
        for name, value in vars(args).items():
            setattr(self, name, value)
       
        self.data = '/media/leelabsg-storage0/wonyoung/leelab/MRI/yAwareContrastiveLearning/adni_t1s_baseline' 
        self.label = './csv/fsdat_baseline.csv' 
        self.label_name = 'Dx.new'  # `Dx.new`
        self.task_type = 'cls'  # 'cls' or 'reg'
        self.valid_ratio = 0.25 # valid set ratio compared to training set
        self.input_size = (1, 80, 80, 80) 
        self.batch_size = 8
        self.pin_mem = True
        self.num_cpu_workers = 8
        self.num_classes = 2 # AD vs CN or MCI vs CN or AD vs MCI or reg
        self.cuda = True

        self.model = 'DenseNet' # 'UNet
        self.nb_epochs = 100
        self.patience = 20
        self.lr = 1e-4 # Optimizer
        self.weight_decay = 5e-5
        self.tf = 'cutout' 
        self.splits = 5
