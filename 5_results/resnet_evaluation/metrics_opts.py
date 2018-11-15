import argparse
import os
import keras.backend as K

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32, help='size of minibatch')
parser.add_argument('--aug_prob', type=float, default=0.0, help='augmentation probability of iaa.Sometimes')
parser.add_argument('--dropout_prob', type=float, default=0.25, help='dropout probability')
parser.add_argument('--neurons', type=int, default=0, help='number of fully_connected_neurons before softmax')
parser.add_argument('--mem_perc', type=float, default=1.0, help='percentage of memory to be reserved for the model')
parser.add_argument('--backend_format', type=str, default='channels_last', help='keras image data format')
parser.add_argument('--fake_ratio', type=float, default=0.0, help='fake/true images ratio')
opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


K.set_image_data_format(opt.backend_format)
