from tqdm import tqdm
import sys,os
filename ='../Eval/list_eval_partition.txt'
def write_part(get_part='train'):
    rst = open('train.lst','w')
    with open(filename,'r') as f:
        for line in f:
            if get_part in line:
                rst.write(line)
    rst.close()
#write_part()
def do_something():
    cnt1=0
    cnt2=0
    cnt3=0
    with open(filename,'r') as f:
        for line in f:
            if 'train' in line:cnt1+=1
            elif 'gallery' in line:cnt2 +=1
            else:cnt3+=1
    # 98832
    print 'train:',cnt1
    # 47773
    print 'test:',cnt2
    print 'val:',cnt3
def do_train():
    rst = open('train.lst','w')
    with open(filename,'r') as f:
        examples = f.readlines()
    for line in tqdm(examples[2:]):
        cur = [a for a in line.strip().split(' ') if a != '']
        if cur[2]=='train':
            rst.write(cur[0]+'\n')
            #rst.write(cur[1]+'\n')
    rst.close()
do_something()
do_train()
