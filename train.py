# coding=utf-8
import argparse
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import  calMetric_iou, LoadDatasetFromFolder
from loss.BCL import BCL
from loss.DiceLoss import DiceLoss
from model.dsamnet import dsamnet
import numpy as np
import random
from train_options import parser

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# set seeds
seed_torch(2020)


if __name__ == '__main__':
    # load data
    train_set = LoadDatasetFromFolder(opt,opt.train1_dir, opt.train2_dir, opt.label_train)
    val_set = LoadDatasetFromFolder(opt,opt.val1_dir, opt.val2_dir, opt.label_val)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.num_workers, batch_size=opt.val_batchsize, shuffle=True)

    # define model
    netCD = dsamnet(opt.n_class).to(device, dtype=torch.float)

    # set optimization
    optimizerCD = optim.Adam(netCD.parameters(),lr= opt.lr, betas=(opt.beta1, 0.999))
    CDcriterion = BCL().to(device, dtype=torch.float)
    CDcriterion1 = DiceLoss().to(device, dtype=torch.float)

    results = {'train_loss': [], 'train_CT':[], 'train_Dice':[],'val_IoU': []}

    # training
    for epoch in range(1, opt.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'Dice_loss':0, 'CT_loss':0, 'CD_loss': 0}

        netCD.train()
        for image1, image2, label in train_bar:
            running_results['batch_sizes'] += opt.batchsize

            image1 = image1.to(device, dtype=torch.float)
            image2 = image2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float) # one-hot label

            prob, ds2, ds3 = netCD(image1, image2)

            #Diceloss
            dsloss2 = CDcriterion1(ds2, label)
            dsloss3 = CDcriterion1(ds3, label)

            Dice_loss = 0.5*(dsloss2+dsloss3)

            # contrative loss
            label = torch.argmax(label, 1).unsqueeze(1).float()
            CT_loss = CDcriterion(prob, label)

            # CD loss
            CD_loss =  CT_loss + opt.wDice *Dice_loss

            netCD.zero_grad()
            CD_loss.backward()
            optimizerCD.step()

            # loss for current batch before optimization
            running_results['Dice_loss'] += Dice_loss.item() * opt.batchsize
            running_results['CT_loss'] += CT_loss.item() * opt.batchsize
            running_results['CD_loss'] += CD_loss.item() * opt.batchsize

            train_bar.set_description(
                desc='[%d/%d] CD: %.4f  ' % (
                    epoch, opt.num_epochs, running_results['CD_loss'] / running_results['batch_sizes'],
                    ))

        # eval
        netCD.eval()

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            inter, unin = 0,0
            valing_results = {'CD_loss': 0, 'batch_sizes': 0, 'Dice_loss':0, 'CT_loss':0,
                              'IoU': 0}

            for image1, image2,  label in val_bar:
                valing_results['batch_sizes'] += opt.val_batchsize

                image1 = image1.to(device, dtype=torch.float)
                image2 = image2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)

                prob, ds2, ds3 = netCD(image1, image2)

                label = torch.argmax(label, 1).unsqueeze(1)

                gt_value = (label > 0).float()
                prob = (prob > 1).float()
                prob = prob.cpu().detach().numpy()
                gt_value = gt_value.cpu().detach().numpy()
                gt_value = np.squeeze(gt_value)
                result = np.squeeze(prob)

                intr, unn = calMetric_iou(gt_value, result)
                inter = inter + intr
                unin = unin + unn

                val_bar.set_description(
                    desc='IoU: %.4f' % (inter * 1.0 / unin))

            valing_results['IoU'] = inter * 1.0 / unin

        # save model parameters
        val_loss = valing_results['IoU']

        if val_loss > 0.5 or epoch==1:
            mloss = val_loss
            torch.save(netCD.state_dict(),  opt.model_dir+'netCD_epoch_%d.pth' % (epoch+opt.pre_num ))

        results['train_loss'].append(running_results['CD_loss'] / running_results['batch_sizes'])
        results['train_CT'].append(running_results['CT_loss'] / running_results['batch_sizes'])
        results['train_Dice'].append(running_results['Dice_loss'] / running_results['batch_sizes'])

        results['val_IoU'].append(valing_results['IoU'])

        if epoch % 1 == 0 :
            data_frame = pd.DataFrame(
                data={'train_loss': results['train_loss'],
                      'val_IoU': results['val_IoU']},
                index=range(1, epoch + 1))
            data_frame.to_csv(opt.sta_dir, index_label='Epoch')
