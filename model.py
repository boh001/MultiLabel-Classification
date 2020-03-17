import os
import cv2
import glob
import dicom
import numpy as np
import pandas as pd
from efficientnet_pytorch import EfficientNet
import torch
from albumentations import Compose, ShiftScaleRotate, Resize
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
import dataset as dt



class TTI(object):
    def __init__(self,args):
        self.epoch = args.epoch
        self.phase = args.phase
        self.model = args.model
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.keep = args.keep
        self.loss = args.loss

        #data_loader
        if self.phase =='train':
            self.transform = transforms.Compose([transforms.ToTensor(),ShiftScaleRotate()])
            self.dataset = dt.K_Dataset(args,csv_file = '/home/boh001/kaggle/train.csv', transform = self.transform)
            self.data_loader = DataLoader(self.dataset,batch_size=self.batch_size, shuffle=False ,num_workers = 5)
            self.inf_dataset = dt.K_Dataset(args,csv_file = '/home/boh001/kaggle/train.csv', transform = self.transform,inf = True)
            self.inf_data_loader = DataLoader(self.inf_dataset,batch_size=self.batch_size, shuffle = False, num_workers = 5)
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.dataset = dt.K_Dataset(args,csv_file = '/home/boh001/kaggle/test.csv',transform = self.transform)
            self.data_loader = DataLoader(self.dataset,batch_size=self.batch_size, shuffle=False ,num_workers = 5)

        #self.train_dataset = dt.K_Dataset(args,csv_file = '/home/boh001/kaggle/train.csv', transform = self.transform_train)
        #self.inf_dataset = dt.K_Dataset(args,csv_file = '/home/boh001/kaggle/train.csv', transform = self.transform_train, inf = True)
        #self.test_dataset = dt.K_Dataset(args,csv_file = '/home/boh001/kaggle/test.csv',transform = self.transform_test, inf = True)
        #self.data_loader = DataLoader(self.dataset,batch_size=self.batch_size, shuffle=True ,num_workers = 5)
        #self.inf_data_loader = DataLoader(self.inf_dataset,batch_size=self.batch_size, shuffle = False, num_workers = 5)
        #self.test_loader = DataLoader(self.test_dataset,batch_size =self.batch_size, shuffle= False,num_workers = 5 )



        #network

        if self.model =='effi':
            self.net = EfficientNet.from_pretrained('efficientnet-b1',num_classes = 6)
            self.net = nn.DataParallel(self.net).cuda()
        else:
            print('아직')

        #loss
        if self.loss == 'wl':
            self.loss = nn.BCEWithLogitsLoss().cuda()
        else:
            print('not yet')

        #optimizer
        if args.optimizer == 'adam':
            if self.model == 'effi':
                self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr)

        else:
            print('안 만들었다 아직')






    def train(self):

        def log_loss(output,label):
          rev_label = (label==0).float()
          output = torch.sigmoid(output)
          loss = torch.abs(output-rev_label)
          loss = -torch.log(loss.prod(-1).mean())

          return loss

        if self.keep>0:
            writer = SummaryWriter(comment="effi",purge_step = self.keep+1)
        else:
            writer = SummaryWriter(comment="effi")


        if self.model == 'effi':
            #lr_sc = lr_scheduler.StepLR(self.optimizer, step_size=1)


            for e in range(self.keep+1,self.epoch+1):
            #continue train
                if self.keep > 0:
                    checkpoint = torch.load('/home/boh001/save_model/effi/{}.pth'.format(self.keep))
                    self.net.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                startTime = time.time()
                filepath = '/home/boh001/save_model//{}/{}.pth'.format(self.model, e)
                total_loss = 0
                total_inf_loss = 0
                acc = 0
                inf_acc = 0

                print('epoch :',e)

                self.net.eval()
                for j,x in enumerate(self.inf_data_loader):

                    with torch.no_grad():

                    #validation loss
                        input = x['image'].float().cuda()
                        target = x['label'].float().cuda()
                        d_input = torch.cat((input,input),1).cuda()
                        input = torch.cat((d_input,input),1).cuda()
                        output = self.net(input).float().cuda()

                        inf_l = log_loss(output,target).cuda()
                        total_inf_loss += inf_l.item()
                        acc_a = (torch.sigmoid(output)>=0.5).float()

                        correct = (acc_a == target).sum()
                        inf_acc += correct*100/(6*self.batch_size)





                self.net.train()
                for i,x in enumerate(self.data_loader):

                    #input 1 channel -> 3 channel
                    input = x['image'].float().cuda()
                    d_input = torch.cat((input,input),1)
                    input = torch.cat((d_input,input),1)

                    #output
                    output = self.net(input).float().cuda()

                    #label
                    target = x['label'].float().cuda()

                    #accuracy
                    with torch.no_grad():
                        acc_a = (torch.sigmoid(output)>=0.5).float()
                        correct = (acc_a == target).sum()
                        acc += correct*100/(6*self.batch_size)

                    #loss
                    l = self.loss(output,target).cuda()
                    total_loss += l.item()

                    #backward
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()


                #lr_sc.step()
                endTime = time.time() - startTime
                print('{} seconds per Epoch :'.format(endTime))

                #tensorboard
                if e % 1 == 0:
                    print(f"Training Loss: {total_loss*self.batch_size/(len(self.data_loader))}")
                    print(f"Training Accuracy: {acc/len(self.data_loader)}%")
                    print(f"Validation Loss: {total_inf_loss*self.batch_size/(len(self.inf_data_loader))}")
                    print(f"Validation Accuracy: {inf_acc/len(self.inf_data_loader)}%")

                    writer.add_scalar('Logger/train_loss', total_loss*self.batch_size/(len(self.data_loader)), e)
                    writer.add_scalar('Logger/train_acc', acc/len(self.data_loader), e)
                    writer.add_scalar('Logger/val_loss', total_inf_loss*self.batch_size/(len(self.inf_data_loader)), e)
                    writer.add_scalar('Logger/val_acc', inf_acc/len(self.inf_data_loader), e)

                    #save
                    torch.save({'model_state_dict':self.net.state_dict(),'optimizer_state_dict':self.optimizer.state_dict()}, filepath)
                    print('Save done : epoch {}'.format(e))
            writer.close()

    def test(self):
        test_pred = np.zeros((len(self.dataset) * 6, 1))
        if self.model == 'effi':
            self.net.eval()

            for i,x in enumerate(self.data_loader):
                #input 1 channel -> 3 channel
                with torch.no_grad():
                    input = x['image'].float().cuda()
                    d_input = torch.cat((input,input),1)
                    input = torch.cat((d_input,input),1)

                    #load path
                    checkpoint = torch.load('/home/boh001/save_model/effi/{}.pth'.format(self.keep))
                    self.net.load_state_dict(checkpoint['model_state_dict'])

                    #output
                    output = self.net(input).cuda()


                    #sigmoid
                    test_pred[(i * self.batch_size * 6):((i + 1) * self.batch_size * 6)] = torch.sigmoid(
                    output).detach().cpu().reshape((len(input) * 6, 1))

            #csv
            sample = pd.read_csv('/data1/kaggle-hemorrhage/stage_1_sample_submission.csv')
            submission =  pd.read_csv('/home/boh001/kaggle/test.csv')
            submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
            submission.columns = ['ID', 'Label']
            submission.reindex(sample.index)
            #a = [b for b in sample.Image]
            #pool = Pool(processes= 40)
            #pool.map(sub, image)

            #def sub():
            #    d = submission[submission.ID.isin([a])].index.tolist()
            #    submission['Label'][d[0]] = test_pred[k]

               #submission.columns = ['ID', 'Label']
            submission.to_csv('submission.csv', index=False)
            print(submission.head())

