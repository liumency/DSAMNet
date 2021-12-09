import argparse

#training options
parser = argparse.ArgumentParser(description='Train Change Detection Models')

# training parameters
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=16, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=24, type=int, help='num_workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')
parser.add_argument('--wDice', type=float, default=0.1, help='weight of dice loss')


# path for loading data
parser.add_argument('--train1_dir', default='../Data/MS_CD/CDD/train/time1', type=str, help='t1 image path for training')
parser.add_argument('--train2_dir', default='../Data/MS_CD/CDD/train/time2', type=str, help='t2 image path for training')
parser.add_argument('--label_train', default='../Data/MS_CD/CDD/train/label', type=str, help='label path for training')

parser.add_argument('--val1_dir', default='../Data/MS_CD/CDD/val/time1', type=str, help='t1 image path for validation')
parser.add_argument('--val2_dir', default='../Data/MS_CD/CDD/val/time2', type=str, help='t2 image path for validation')
parser.add_argument('--label_val', default='../Data/MS_CD/CDD/val/label', type=str, help='label path for validation')


# network saving and loading parameters
parser.add_argument('--model_dir', default='epochs/CDD/', type=str, help='save path for CD model')
parser.add_argument('--sta_dir', default='statistics/CDD.csv', type=str, help='statistics')








