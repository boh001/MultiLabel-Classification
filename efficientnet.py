import os
import cv2
import glob
import dicom
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, Resize, HorizontalFlip, VerticalFlip, RandomBrightnessContrast
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensor
from sklearn.metrics import log_loss
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from optimizers import RAdam
import argparse
import time
from sklearn.model_selection import train_test_split
from PIL import Image
from ipywidgets import IntProgress
#multiprocessing

parser = argparse.ArgumentParser(description='')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument('--use_cutout', action='store_true', default=False)
parser.add_argument('--cutout_size', type=int, default=16)
parser.add_argument('--cutout_prob', type=float, default=1)
parser.add_argument('--cutout_inside', action='store_true', default=False)
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--model', dest='model', default='b1', help='effi_model')
parser.add_argument('--keep', dest='keep', type=int, default=0, help='continue training')
parser.add_argument('--dir_csv', dest='dir_csv', default='/data1/kaggle-hemorrhage/', help='dicom file directory')
parser.add_argument('--batch_size', dest='batch_size', type = int, default=150, help='dicom file directory')
parser.add_argument('--n_classes', dest='n_classes', default=6, help='dicom file directory')
parser.add_argument('--n_epochs', dest='n_epochs', default=10, help='dicom file directory')
#parser.add_argument('--dir_train_img', dest='dir_train_img', default='/data1/kaggle-hemorrhage/stage_1_train_images_png/', help='train image folder name')
#parser.add_argument('--dir_test_img', dest='dir_test_img', default='/data1/kaggle-hemorrhage/stage_1_test_images_png/', help='test image folder name')
parser.add_argument('--dir_train_img', dest='dir_train_img', default='/home/boh001/image/kaggle/train/', help='train image folder name')
parser.add_argument('--dir_test_img', dest='dir_test_img', default='/home/boh001/image/kaggle/test/', help='test image folder name')
parser.add_argument('--extension', dest='extension', default= 'dcm', help='extension, [IMA, DCM]')
parser.add_argument('--tensorboardwriter', dest='tensorboardwriter', default= 'exp2', help='extension, [IMA, DCM]')
parser.add_argument('--csvname', dest='csvname', default='submission.csv', help='name')
args = parser.parse_args()


class IntracranialDataset(Dataset):
    def __init__(self,csv_file, path, labels, transform=None):
        self.path = path
        self.data = pd.read_csv(csv_file)[:100]
        self.transform = transform
        self.labels = labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        def cutout(image, mask_size, p, cutout_inside, mask_color=0):
            mask_size_half = mask_size // 2
            offset = 1 if mask_size % 2 == 0 else 0
            image = np.asarray(image).copy()
            d = len(image)
            if np.random.random() > p:
                return image

            h, w = image.shape[1], image.shape[2]
            if cutout_inside:
                cxmin, cxmax = mask_size_half, w + offset - mask_size_half
                cymin, cymax = mask_size_half, h + offset - mask_size_half
            else:
                cxmin, cxmax = 0, w + offset
                cymin, cymax = 0, h + offset

            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
            xmin = cx - mask_size_half
            ymin = cy - mask_size_half
            xmax = xmin + mask_size
            ymax = ymin + mask_size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            for j in range(3):
                image[j][ymin:ymax, xmin:xmax] = mask_color
            return image

        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')
        img = cv2.imread(img_name)

        if self.transform:
            augmented = self.transform(image=img)
            #img = cutout(augmented['image'],args.cutout_size, args.cutout_prob, args.cutout_inside)
            img = augmented['image']


        if self.labels:
            labels = torch.tensor(self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])

            return {'image': img, 'labels': labels}

        else:

            return {'image': img}

# Data loaders



transform_train = Compose([
    ShiftScaleRotate(),
    HorizontalFlip(p=0.5,),
    VerticalFlip(p=0.5,),
    RandomBrightnessContrast(0.08,0.08),
    ToTensor()
])

transform_validation = Compose([
    ToTensor()
])
transform_test= Compose([
    ToTensor()
])

train = pd.read_csv(os.path.join(args.dir_csv, 'stage_1_train.csv'))

test = pd.read_csv(os.path.join(args.dir_csv, 'stage_1_sample_submission.csv'))

train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']
train.head()

# Some files didn't contain legitimate images, so we need to remove them
png = glob.glob(os.path.join(args.dir_train_img, '*.png'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)
train = train[train['Image'].isin(png)]

X_train, X_validation = train_test_split(train, test_size=0.2, random_state=123)

X_train.to_csv('train.csv', index=False)
X_validation.to_csv('validation.csv', index=False)

# Also prepare the test data

test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)
test['Image'] = 'ID_' + test['Image']
test = test[['Image', 'Label']]
test.drop_duplicates(inplace=True)

test.to_csv('test.csv', index=False)


if args.phase == 'train':
    dataset = IntracranialDataset(
        csv_file='train.csv', path=args.dir_train_img, transform=transform_train,labels=True)

    validation_dataset = IntracranialDataset(
        csv_file='validation.csv', path=args.dir_train_img, transform=transform_validation,labels=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=10)

    data_loader_valid = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=10)
elif args.phase == 'test':
    test_dataset = IntracranialDataset(
        csv_file='test.csv', path=args.dir_test_img, transform=transform_test, labels=False)



    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

# Model
if args.model == 'b2':
    model = EfficientNet.from_pretrained('efficientnet-b2',num_classes = 6)
elif args.model == 'b0':
    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes = 6)
elif args.model == 'b1':
    model = EfficientNet.from_pretrained('efficientnet-b1',num_classes = 6)

elif args.model == 'b4':
    model = EfficientNet.from_pretrained('efficientnet-b4',num_classes = 6)
elif args.model == 'b5':
    model = EfficientNet.from_pretrained('efficientnet-b5',num_classes = 6)

model = torch.nn.DataParallel(model).cuda()
#model.to(device)

criterion = torch.nn.BCEWithLogitsLoss().cuda()
sub_criterion = torch.nn.BCEWithLogitsLoss(reduce=False,reduction=None).cuda()

plist = [{'params': model.parameters(), 'lr': 0.00001}]
optimizer = RAdam(plist, lr=0.00001)


def log_loss(output, label):
    rev_label = (label == 0).float()
    output = torch.sigmoid(output)
    loss = torch.abs(output - rev_label)
    loss = -torch.log(loss.prod(-1).mean())
    return loss


# Train
#lr_sc = lr_scheduler.StepLR(optimizer, step_size=2)
if args.keep > 0:
    checkpoint = torch.load('/home/boh001/save_model/effi/{}.pth'.format(args.keep))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
writer = SummaryWriter(comment="effi")

for epoch in range(args.keep+1,args.n_epochs+1):

    startTime = time.time()
    print('Epoch {}/{}'.format(epoch, args.n_epochs))
    print('-' * 10)

    model.train()
    tr_loss = 0
    sub_tr_loss = 0
    val_loss = 0
    sub_val_loss = 0
   # tk0 = tqdm(data_loader, desc="Iteration")
    for step, batch in enumerate(data_loader):
        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(dtype=torch.float).cuda()
        labels = labels.to(dtype=torch.float).cuda()
        outputs = model(inputs).cuda()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()


        loss.backward()
        optimizer.step()
       # lr_sc.step()
        with torch.no_grad():
            sub_loss = sub_criterion(outputs, labels).mean(dim=-2)
            tr_loss += loss.item()
            sub_tr_loss += sub_loss
       # print('batch Loss: {:.4f}'.format(loss))
    epoch_loss = tr_loss*args.batch_size / len(data_loader)
    epoch_sub_loss = sub_tr_loss * args.batch_size / len(data_loader)
    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict()}, os.path.join('/home/boh001/save_model/effi', '{}.pth'.format(epoch)))
    print('Training Loss: {:.4f}'.format(epoch_loss))
    print('epidural: {:.4f}'.format(epoch_sub_loss[0]))
    print('intraparenchymal: {:.4f}'.format(epoch_sub_loss[1]))
    print('intraventricular: {:.4f}'.format(epoch_sub_loss[2]))
    print('subarachnoid: {:.4f}'.format(epoch_sub_loss[3]))
    print('subdural: {:.4f}'.format(epoch_sub_loss[4]))
    print('any: {:.4f}'.format(epoch_sub_loss[5]))

    model.eval()

    #tk1 = tqdm(data_loader_valid, desc='iteration')
    with torch.no_grad():
        for step, batch in enumerate(data_loader_valid):
            inputs = batch["image"]
            labels = batch["labels"]

            inputs = inputs.to(dtype=torch.float).cuda()
            labels = labels.to(dtype=torch.float).cuda()
            outputs = model(inputs)

            loss = log_loss(outputs, labels)
            sub_loss = sub_criterion(outputs,labels).mean(dim=-2)
            val_loss += loss.item()
            sub_val_loss += sub_loss

          #  print('batch Loss: {:.4f}'.format(val_loss))
        epoch_val_loss = args.batch_size*val_loss/len(data_loader_valid)
        epoch_sub_val_loss = args.batch_size * sub_val_loss / len(data_loader_valid)

    endTime = time.time() - startTime
    print('{} seconds per Epoch :'.format(endTime))
    print('Validation Loss: {:.4f}'.format(epoch_val_loss))
    print('epidural: {:.4f}'.format(epoch_sub_val_loss[0]))
    print('intraparenchymal: {:.4f}'.format(epoch_sub_val_loss[1]))
    print('intraventricular: {:.4f}'.format(epoch_sub_val_loss[2]))
    print('subarachnoid: {:.4f}'.format(epoch_sub_val_loss[3]))
    print('subdural: {:.4f}'.format(epoch_sub_val_loss[4]))
    print('any: {:.4f}'.format(epoch_sub_val_loss[5]))
    #train
    writer.add_scalar('Loss/train_loss', epoch_loss, epoch)
    writer.add_scalars('Loss/Subtype', {
        'epidural': epoch_sub_loss[0],
        'intraparenchymal': epoch_sub_loss[1],
        'intraventricular': epoch_sub_loss[2],
        'subarachnoid': epoch_sub_loss[3],
        'subdural': epoch_sub_loss[4],
        'any': epoch_sub_loss[5]}, epoch)
    #writer.add_scalar('Loss/Subtype/epidural', epoch_sub_loss[0], epoch)
    #writer.add_scalar('Loss/Subtype/intraparenchymal', epoch_sub_loss[1], epoch)
    #writer.add_scalar('Loss/Subtype/intraventricular', epoch_sub_loss[2], epoch)
    #writer.add_scalar('Loss/Subtype/subarachnoid', epoch_sub_loss[3], epoch)
    #writer.add_scalar('Loss/Subtype/subdural', epoch_sub_loss[4], epoch)
    #writer.add_scalar('Loss/Subtype/any', epoch_sub_loss[5], epoch)
    # vali
    writer.add_scalar('VLoss/val_loss', epoch_val_loss, epoch)
    writer.add_scalars('VLoss/Subtype',{
        'epidural':epoch_sub_val_loss[0],
        'intraparenchymal':epoch_sub_val_loss[1],
        'intraventricular':epoch_sub_val_loss[2],
        'subarachnoid':epoch_sub_val_loss[3],
        'subdural':epoch_sub_val_loss[4],
        'any':epoch_sub_val_loss[5]}, epoch)
    #writer.add_scalar('VLoss/Subtype/intraparenchymal', epoch_sub_val_loss[1], epoch)
    #writer.add_scalar('VLoss/Subtype/intraventricular', epoch_sub_val_loss[2], epoch)
    #writer.add_scalar('VLoss/Subtype/subarachnoid', epoch_sub_val_loss[3], epoch)
    #writer.add_scalar('VLoss/Subtype/subdural', epoch_sub_val_loss[4], epoch)
    #writer.add_scalar('VLoss/Subtype/any', epoch_sub_val_loss[5], epoch)
writer.close()

if args.phase == 'test':

    model.load_state_dict(torch.load('/home/boh001/save_model/effi/{}.pth'.format(args.keep)))
    model.eval()
    test_pred = np.zeros((len(test_dataset) * args.n_classes, 1))

    for i, x_batch in enumerate(tqdm(data_loader_test)):
        x_batch = x_batch["image"]
        x_batch = x_batch.to(dtype=torch.float).cuda()

        with torch.no_grad():
            pred = model(x_batch)

            test_pred[(i * args.batch_size * args.n_classes):((i + 1) * args.batch_size * args.n_classes)] = torch.sigmoid(
                pred).detach().cpu().reshape((len(x_batch) * args.n_classes, 1))

    submission = pd.read_csv(os.path.join(args.dir_csv, 'stage_1_sample_submission.csv'))
    submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
    submission.columns = ['ID', 'Label']

    submission.to_csv(args.csvname, index=False)
    submission.head()
