#!/usr/bin/python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import os
from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.

hengs_path = 'hengcodes'
sys.path.append(hengs_path)

from common  import *
from model   import *
from augmentation import *
from kagglemetric import *

data_root = "./train"
df = pd.read_csv("train.csv")
train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=2411)

TASK_NAME = [ 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic' ]
NUM_TASK = len(TASK_NAME)


def to_64x112(image):
    image = cv2.resize(image,dsize=(128,128),interpolation=cv2.INTER_AREA)
    #image = cv2.resize(image,dsize=None, fx=64/137,fy=64/137,interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image,0,0,1,1,cv2.BORDER_CONSTANT,0)
    return image

def compute_kaggle_metric_by_decode(probability, truth, weight=[2,1,1]):
    predict = [ p.argmax(-1) for p in probability ]

    def compute_recall(predict, truth, num_class):
        correct = predict==truth
        recall = np.zeros(num_class)
        for c in range(num_class):
            e = correct[truth==c]
            if len(e)>0:
                recall[c]=e.mean()
        return recall

#    with open("inference.dump.txt", "w") as fd:
#        for i in range(3):
#            for j in range(len(predict[i])):
#                (p, t) = (predict[i][j], truth[i][j])
#                print("class=%d,image=%s,pred=%d,truth=%d" % (i, valid_df.iloc[j]["image_id"], p, t), file=fd)

    recall   = [ compute_recall(predict[i],truth[i],NUM_CLASS[i]) for i in range(3) ]
    componet = [ r.mean() for r in recall ]

    average = np.average(componet, weights=weight)
    return average, componet, recall


class KaggleDataset(Dataset):
    def __init__(self, df, data_path, augment=None):
        self.image_ids = df['image_id'].values
        self.grapheme_roots = df['grapheme_root'].values
        self.vowel_diacritics = df['vowel_diacritic'].values
        self.consonant_diacritics = df['consonant_diacritic'].values

        self.data_path = data_path
        self.augment = augment

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        return string


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, index):
        # print(index)
        image_id = self.image_ids[index]
        grapheme_root = self.grapheme_roots[index]
        vowel_diacritic = self.vowel_diacritics[index]
        consonant_diacritic = self.consonant_diacritics[index]

        image_id = os.path.join(self.data_path, image_id + '.png')

        image = cv2.imread(image_id, 0)
        image = image.reshape(137, 236)
        #image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        image = image.astype(np.float32)/255
        label = [grapheme_root, vowel_diacritic, consonant_diacritic]

        infor = Struct(
            index    = index,
            image_id = image_id,
        )

        if self.augment is None:
            return image, label, infor
        else:
            return self.augment(image, label, infor)

def null_collate(batch):
    batch_size = len(batch)

    input = []
    label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        label.append(batch[b][1])
        infor.append(batch[b][-1])

    label = np.stack(label)

    input = np.stack(input)
    #input = input[...,::-1].copy()
    #input = input.transpose(0,3,1,2)

    #----
    input = torch.from_numpy(input).float()
    input = input.unsqueeze(1) #.repeat(1,3,1,1) # 3 common

    truth = torch.from_numpy(label).long()
    truth = torch.unbind(truth,1)
    return input, truth, infor


def train_collate(batch):
    batch_size = len(batch)
    original = []
    input = []
    label = []
    infor = []
    for b in range(batch_size):
        original.append(batch[b][0])
        input.append(batch[b][1])
        label.append(batch[b][2])
        infor.append(batch[b][-1])

    label = np.stack(label)
    input = np.stack(input)
    original = np.stack(original)

    #----
    input = torch.from_numpy(input).float().unsqueeze(1)
    original = torch.from_numpy(original).float().unsqueeze(1)

    truth = torch.from_numpy(label).long()
    truth = torch.unbind(truth,1)
    return original, input, truth, infor

def train_augment(image, label, infor):
    original = image.copy()
    original = to_64x112(original)
    if 1:
        for op in np.random.choice([
            lambda image : do_identity(image),
            lambda image : do_random_projective(image, 0.4),
            lambda image : do_random_perspective(image, 0.4),
            lambda image : do_random_scale(image, 0.4),
            lambda image : do_random_rotate(image, 0.4),
            lambda image : do_random_shear_x(image, 0.5),
            lambda image : do_random_shear_y(image, 0.4),
            lambda image : do_random_stretch_x(image, 0.5),
            lambda image : do_random_stretch_y(image, 0.5),
            lambda image : do_random_grid_distortion(image, 0.4),
            lambda image : do_random_custom_distortion1(image, 0.5),
        ],1):
            image = op(image)

        for op in np.random.choice([
            lambda image : do_identity(image),
            lambda image : do_random_erode(image, 0.4),
            lambda image : do_random_dilate(image, 0.4),
            lambda image : do_random_sprinkle(image, 0.5),
            lambda image : do_random_line(image, 0.5),
        ],1):
            image = op(image)

        for op in np.random.choice([
            lambda image : do_identity(image),
            lambda image : do_random_contast(image, 0.5),
            lambda image : do_random_block_fade(image, 0.5),
        ],1):
            image = op(image)

        image = do_random_pad_crop(image, 3)
        image = to_64x112(image)
    return original, image, label, infor

def train_batch_augment(original, input, onehot):
    if 1:
        operation = [
            #lambda input, onehot : (input, onehot, None),
            #lambda input, onehot : do_random_batch_mixup(original, onehot),
            #lambda input, onehot : do_random_batch_cutout(original, onehot),

            lambda input, onehot : (input, onehot, None),
            lambda input, onehot : do_random_batch_mixup(original, onehot),
            lambda input, onehot : do_random_batch_cutmix(original, onehot),
            lambda input, onehot : do_random_batch_cutout(original, onehot),
        ]
        #op = np.random.choice(operation, p=[0.4,0.1,0.1,0.4])
        op = np.random.choice(operation, p=[0.1,0.425,0.425,0.05])
        #op = np.random.choice(operation, p=[0.30,0.20,0.20,0.30])
        #op = np.random.choice(operation, p=[0.50,0.25,0.25])
        #op = np.random.choice(operation)

        with torch.no_grad():
            input, onehot, perm = op(input, onehot)

    return input, onehot

def valid_augment(image, label, infor):
    image = to_64x112(image)
    return image, label, infor

def do_valid(net, valid_loader, out_dir=None):

    valid_loss = np.zeros(6, np.float32)
    valid_num  = np.zeros_like(valid_loss)

    valid_probability = [[],[],[],]
    valid_truth = [[],[],[],]

    for t, (input, truth, infor) in enumerate(valid_loader):

        #if b==5: break
        batch_size = len(infor)

        net.eval()
        input  = input.cuda()
        truth  = [t.cuda() for t in truth]
        onehot = [to_onehot(t,c) for t,c in zip(truth,NUM_CLASS)]

        with torch.no_grad():
            logit, feature = data_parallel(net, input) #net(input)
            probability = logit_to_probability(logit)

            loss = criterion(logit, onehot)
            correct = metric(probability, truth)

        #---
        loss = [l.item() for l in loss]
        l = np.array([ *loss, *correct, ])*batch_size
        n = np.array([ 1, 1, 1, 1, 1, 1  ])*batch_size
        valid_loss += l
        valid_num  += n

        #---
        for i in range(3):
            valid_probability[i].append(probability[i].data.cpu().numpy())
            valid_truth[i].append(truth[i].data.cpu().numpy())

        #print(valid_loss)
        print('\r %8d / %d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)
    assert(valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss/(valid_num+1e-8)

    #------
    for i in range(3):
        valid_probability[i] = np.concatenate(valid_probability[i])
        valid_truth[i] = np.concatenate(valid_truth[i])
    #average, componet, recall = compute_kaggle_metric(valid_probability, valid_truth)

    average, componet, recall = compute_kaggle_metric_by_decode(valid_probability, valid_truth)

    return valid_loss, (average, componet, recall)


def run_train():
    out_dir = 'working'
    initial_checkpoint = '/mnt/ssd/bengalikaggle/working/checkpoint/00090000_model.pth'
    #initial_checkpoint = '/mnt/ssd/bengalikaggle/resnet50_full.working/checkpoint/00150000_model.pth'
    #initial_checkpoint = None

    schduler = NullScheduler(lr=0.008) #0.005)
    iter_accum = 1
    batch_size = 96 # 28 # 96 #64 #8

    ## setup  -----------------------------------------------------------------------------
    for f in ['checkpoint','train','valid'] : os.makedirs(out_dir +'/'+f, exist_ok=True)
        
    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = KaggleDataset(
        df = train_df, 
        data_path = data_root,
        augment = train_augment,
    )
    train_loader  = DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = train_collate
    )


    valid_dataset = KaggleDataset(
        df = valid_df, 
        data_path = data_root,
        augment = valid_augment,
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler     = SequentialSampler(valid_dataset),
        batch_size  = batch_size,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        # for k in list(state_dict.keys()):
        #      if any(s in k for s in ['logit',]): state_dict.pop(k, None)
        # net.load_state_dict(state_dict,strict=False)

        net.load_state_dict(state_dict,strict=True)  #True
    else:
        #net.load_pretrain(pretrain_file = 'pytorch-pretrained-models/densenet121-a639ec97.pth', is_print=False)
        net.load_pretrain(is_print=False)


    log.write('net=%s\n'%(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.05) #,lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0),
                                nesterov=False, momentum=0.5, weight_decay=0.00001)

    num_iters   = 3000*1000 # use this for training longer
    iter_smooth = 50
    iter_log    = 250
    iter_valid  = 500
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 10000))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('                    |----------------------- VALID------------------------------------|------- TRAIN/BATCH -----------\n')
    log.write('rate    iter  epoch | kaggle                    | loss               acc              | loss             | time       \n')
    log.write('----------------------------------------------------------------------------------------------------------------------\n')
              #0.01000  26.2  15.1 | 0.971 : 0.952 0.992 0.987 | 0.22, 0.07, 0.07 : 0.94, 0.98, 0.98 | 0.37, 0.13, 0.13 | 0 hr 13 min

    def message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'):
        #print([rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode])
        if mode==('print'):
            asterisk = ' '
            loss = batch_loss
        if mode==('log'):
            asterisk = '*' if iter in iter_save else ' '
            loss = train_loss

        text = '%0.5f %5.1f%s %4.1f | '%(rate, iter/1000, asterisk, epoch,)
        text = text + '%0.3f : %0.3f %0.3f %0.3f | '%(kaggle[0],*kaggle[1])
        text = text + '%4.2f, %4.2f, %4.2f : %4.2f, %4.2f, %4.2f | '%(*valid_loss,)
        text = text + '%4.2f, %4.2f, %4.2f |'%(*loss,)
        text = text + '%s' % (time_to_str((timer() - start_timer),'min'))

        return text

    #----
    kaggle = (0,0,0,0)
    valid_loss = np.zeros(6,np.float32)
    train_loss = np.zeros(3,np.float32)
    batch_loss = np.zeros_like(train_loss)
    iter = 0
    i    = 0



    start_timer = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        optimizer.zero_grad()
        for t, (original, input, truth, infor) in enumerate(train_loader):

            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch
            
            #if 0:
            if (iter % iter_valid==0):
                valid_loss, kaggle = do_valid(net, valid_loader, out_dir) #
                #print(kaggle)
#                recall = kaggle[2]
#                for i in range(3):
#                    for j in range(NUM_CLASS[i]):
#                        print("recall: %d,%03d,%f" % (i, j, recall[i][j]))
                pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                log.write(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='log'))
                log.write('\n')

            #if 0:
            if iter in iter_save:
                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                if iter!=start_iter:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    pass

            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            #net.set_mode('train',is_freeze_bn=True)

            original = original.cuda()
            input  = input.cuda()
            truth  = [t.cuda() for t in truth]
            onehot = [to_onehot(t,c) for t,c in zip(truth,NUM_CLASS)]

            input, onehot = train_batch_augment(original, input, onehot)

            net.train()
            logit, feature = data_parallel(net, input)
            probability = logit_to_probability(logit)
            loss = criterion(logit, onehot)

            (( 2*loss[0]+loss[1]+loss[2] )/iter_accum).backward()

            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  --------
            loss = [l.item() for l in loss]
            l = np.array([ *loss, ])*batch_size
            n = np.array([ 1, 1, 1 ])*batch_size
            batch_loss      = l/(n+1e-8)
            sum_train_loss += l
            sum_train      += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum_train+1e-12)
                sum_train_loss[...] = 0
                sum_train[...]      = 0


            print('\r',end='',flush=True)
            print(message(rate, iter, epoch, kaggle, valid_loss, train_loss, batch_loss, mode='print'), end='',flush=True)
            i=i+1

        pass  #-- end of one data loader --
    pass #-- end of all iterations --
    
    log.write('\n')


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_train()
