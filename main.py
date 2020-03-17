import argparse
import os
import sys
import model as md

parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_input_path', dest='train_input_path', default='/data1/kaggle-hemorrhage/stage_1_train_images_png', help='LDCT image folder name')
parser.add_argument('--test_input_path', dest='test_input_path', default='/data1/kaggle-hemorrhage/stage_1_test_images_png', help='LDCT image folder name')

#parser.add_argument('--train_input_path', dest='train_input_path', default='/home/boh001/image/kaggle/train', help='LDCT image folder name')
#parser.add_argument('--test_input_path', dest='test_input_path', default='/home/boh001/image/kaggle/test', help='LDCT image folder name')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--model', dest='model', default='effi', help='model')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='epoch')
parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='lr')
parser.add_argument('--loss', dest = 'loss', default = 'wl', help = 'loss')
parser.add_argument('--optimizer', dest = 'optimizer', default= 'adam', help = 'optimizer')
parser.add_argument('--train_ratio', dest='train_ratio', type=float, default=0.8, help='# of train samples')
#parser.add_argument('--seed', dest='seed', type=int, default=0, help='ramdom sampling seed num(for train/test)')
parser.add_argument('--keep', dest='keep', type=int, default=0, help='continue training')
# -------------------------------------
args = parser.parse_args()

model = md.TTI(args)

if args.phase == 'train':
    model.train()
else:
    model.test()

