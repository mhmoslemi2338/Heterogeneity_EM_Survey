import numpy as np
import pandas as pd

import os
import json
import torch
import numpy as np
import random
from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from torch.utils import data
from ditto_light.ditto import DittoModel
import pandas as pd
import csv
import nltk
nltk.download('words')
from nltk.corpus import words
word_list = words.words()
import random
from ditto_light.ditto import train

NUM =0
gpu_no = str(NUM)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


def dynamic_convert_csv_to_txt(input_csv, output_txt):
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_txt, mode='w', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        for row in reader:
            parts = {'left': '', 'right': ''}
            for col_name, value in row.items():
                if '_' in col_name:
                    prefix, attribute = col_name.rsplit('_', 1)
                    parts[prefix] += f"COL {attribute} VAL {value} "
            
            left_part = parts.get('left', '').strip()
            right_part = parts.get('right', '').strip()
            label = row.get('label', '').strip()
            
            outfile.write(f"{left_part} \t{right_part} \t{label}\n")












import random
import torch

for task in ['Fodors-Zagats','iTunes-Amazon']:#,'Walmart-Amazon']:
    for frac in [0.05,0.1,0.15,0.2,0.25]:



        for rep in range(1,3):


            dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
            embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 


            df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/train.csv')
            num_to_flip = int(len(df) * frac)
            indices_to_flip = np.random.choice(df.index, size=num_to_flip, replace=False)
            df.loc[indices_to_flip, 'label'] = 1 - df.loc[indices_to_flip, 'label']
            df.to_csv(dataset_dir+'train2.csv',index=False)
            


            df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/valid.csv')
            num_to_flip = int(len(df) * frac)
            indices_to_flip = np.random.choice(df.index, size=num_to_flip, replace=False)
            df.loc[indices_to_flip, 'label'] = 1 - df.loc[indices_to_flip, 'label']
            df.to_csv(dataset_dir+'vlaid2.csv',index=False)



   


            trainset = "/home/mmoslem3/DATA_VLDB/" + task + '/train2.txt'
            validset = "/home/mmoslem3/DATA_VLDB/" + task+ '/valid2.txt'
            testset = "/home/mmoslem3/DATA_VLDB/" + task +'/test.txt'


            
            dynamic_convert_csv_to_txt(trainset.replace('txt','csv'), trainset)
            dynamic_convert_csv_to_txt(validset.replace('txt','csv'), validset)
            dynamic_convert_csv_to_txt(testset.replace('txt','csv'), testset)



            data_dir = dataset_dir
            run_id = 0
            batch_size = 16
            max_len = 256
            lr = 1e-5
            n_epochs = 100
            alpha_aug = 0.8
            size = None
            lm = 'distilbert'
            da = None # default=None
            dk = None # default=None




            logdir = "checkpoints/"
            summarize = True
            finetuning = True
            save_model = True
            fp16 = False




            configs={task:{"name": task, "task_type": "classification", "vocab": ["0", "1"], 
                            "trainset": "/home/mmoslem3/DATA_VLDB/" + task+ '/train2.txt',
                            "validset": "/home/mmoslem3/DATA_VLDB/" + task+ '/valid2.txt',
                            "testset": "/home/mmoslem3/DATA_VLDB/" + task+ "/test.txt",}, }

            config = configs[task]
            trainset = config['trainset']
            validset = config['validset']
            testset = config['testset']


            # # summarize the sequences up to the max sequence length
            summarizer = Summarizer(config, lm=lm)
            trainset = summarizer.transform_file(trainset, max_len=max_len)
            validset = summarizer.transform_file(validset, max_len=max_len)
            testset = summarizer.transform_file(testset, max_len=max_len)


            # load train/dev/test sets
            train_dataset = DittoDataset(trainset,
                                            lm=lm,
                                            max_len=max_len,
                                            size=size,
                                            da=da)
            valid_dataset = DittoDataset(validset, lm=lm)
            test_dataset = DittoDataset(testset, lm=lm)


            HP = {} 
            HP['n_epochs'] = n_epochs
            HP['batch_size'] = batch_size
            HP['logdir'] = logdir
            HP['n_epochs'] = n_epochs
            HP['save_model'] = save_model
            HP['logdir'] = os.path.join(logdir, task+str(rep)+'hier'+str(100*frac+1))
            HP['task'] = task
            HP['alpha_aug'] = alpha_aug
            HP['fp16'] = fp16
            HP['lm'] = lm
            HP['lr'] = lr

            run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, lm, da,dk, summarize, str(size), run_id)
            run_tag = run_tag.replace('/', '_')


            device = 'cuda:'+str(NUM) if torch.cuda.is_available() else 'cpu'
            train(train_dataset,
                    valid_dataset,
                    test_dataset,
                    run_tag, HP)


            

            model = DittoModel(device=device,lm=HP['lm'],alpha_aug=HP['alpha_aug'])

            model.to(device)
            # directory = HP['logdir']
            ckpt_path = os.path.join(os.path.join(logdir, task+str(rep)+'hier'+str(100*frac+1)), 'model.pt')
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model'])



            torch.cuda.empty_cache()
            frac2 = str(int(frac*100))
            if len(frac2) == 1: frac2 = frac2 +'0'





            testset = summarizer.transform_file(testset, max_len=max_len)
            test_dataset = DittoDataset(testset, lm=lm)


            test_iter = data.DataLoader(dataset=test_dataset,
                                            batch_size=HP['batch_size'],
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=train_dataset.pad)





            all_y = []
            all_probs = []
            with torch.no_grad():
                for  batch in test_iter:
                    x, y = batch
                    logits = model(x)
                    probs = logits.softmax(dim=1)[:, 1]
                    all_probs += probs.cpu().numpy().tolist()
                    all_y += y.cpu().numpy().tolist()
                    

            frac2 = str(int(frac*100))
            if len(frac2) == 1: frac2 = frac2 +'0'


            try: 
                os.mkdir('/home/mmoslem3/RES_hierarch/DITTO_'+task+'_'+'lblnoise')
            except:
                pass
            df = pd.concat([pd.DataFrame(all_probs, columns=['score']),pd.DataFrame(all_y, columns=['true'])], axis=1)            

            df.to_csv('/home/mmoslem3/RES_hierarch/DITTO_'+task+'_'+'lblnoise'+ '/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
            print(task + '_'+str(frac2)+'_'+str(rep))





