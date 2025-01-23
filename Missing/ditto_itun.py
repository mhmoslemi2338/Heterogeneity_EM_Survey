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

gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
NUM =1

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

import numpy as np
import pandas as pd
import time






def clac_p_berno(val):
    MISS_COND = {1 : 0.8, 0: 0.2}
    x = MISS_COND[val['label']]
    return (np.arctan(x)/np.arctan(len(MISS_COND)))+ random.uniform(-0.05, 0.05)



def inject_MCAR(df_in,frac,cols):
    df = df_in.copy()

    n_missing = int(frac * len(df) * len(cols))
    missing_indices = []
    while len(missing_indices) < n_missing:
        row = np.random.randint(0, df.shape[0])
        col = np.random.randint(0, len(cols))
        missing_indices.append(np.array([row, col]))
        missing_indices  = list(np.unique(missing_indices,axis = 0))
    for row, col in missing_indices:
        df.loc[row,'left_'+cols[col]] = np.nan

    n_missing = int(frac * len(df) * len(cols))
    missing_indices = []
    while len(missing_indices) < n_missing:
        row = np.random.randint(0, df.shape[0])
        col = np.random.randint(0, len(cols))
        missing_indices.append(np.array([row, col]))
        missing_indices  = list(np.unique(missing_indices,axis = 0))
    for row, col in missing_indices:
        df.loc[row,'right_'+cols[col]] = np.nan
    return df



def inject_MAR(df_in,frac,cols):
    df = df_in.copy()
    left_cols = ['left_'+col for col in cols]
    right_cols = ['right_'+col for col in cols]

    n_missing = int(frac * len(df) * len(cols))
    number_nans = df.isna()[left_cols].sum().sum()
    while number_nans < n_missing:
        col = np.random.randint(0, len(cols))
        row = np.random.randint(0, df.shape[0])
        p_berno  = clac_p_berno(df.loc[row])
        action = [np.random.rand() < p_berno for _ in range(1)][0]
        if action : 
            df.loc[row,'left_'+cols[col]] = np.nan
        number_nans = df.isna()[left_cols].sum().sum()

    n_missing = int(frac * len(df) * len(cols))
    number_nans = df.isna()[right_cols].sum().sum()
    while number_nans < n_missing:
        col = np.random.randint(0, len(cols))
        row = np.random.randint(0, df.shape[0])
        p_berno  = clac_p_berno(df.loc[row])
        action = [np.random.rand() < p_berno for _ in range(1)][0]
        if action : 
            df.loc[row,'right_'+cols[col]] = np.nan
        number_nans = df.isna()[right_cols].sum().sum()

    return df








def clac_p_berno_MNAR(val,COL):
    MISS_COND = {1 : 0.8, 0: 0.2}
    x = MISS_COND[val['label']] 
    x+= (hash(val[COL])% 100)/100
    return (np.arctan(x)/np.arctan(len(MISS_COND)+100)) + random.uniform(-0.05, 0.05)


def inject_MNAR(df_in,frac,cols):
    df = df_in.copy()
    left_cols = ['left_'+col for col in cols]
    right_cols = ['right_'+col for col in cols]

    n_missing = int(frac * len(df) * len(cols))
    number_nans = df.isna()[left_cols].sum().sum()
    while number_nans < n_missing:
        col = np.random.randint(0, len(cols))
        row = np.random.randint(0, df.shape[0])
        p_berno  = clac_p_berno_MNAR(df.loc[row],'left_'+cols[col])
        action = [np.random.rand() < p_berno for _ in range(1)][0]
        if action : 
            df.loc[row,'left_'+cols[col]] = np.nan
        number_nans = df.isna()[left_cols].sum().sum()
    
    n_missing = int(frac * len(df) * len(cols))
    number_nans = df.isna()[right_cols].sum().sum()
    while number_nans < n_missing:
        col = np.random.randint(0, len(cols))
        row = np.random.randint(0, df.shape[0])
        p_berno  = clac_p_berno_MNAR(df.loc[row],'right_'+cols[col])
        action = [np.random.rand() < p_berno for _ in range(1)][0]
        if action : 
            df.loc[row,'right_'+cols[col]] = np.nan
        number_nans = df.isna()[right_cols].sum().sum()
    
    return df







for task in ['Fodors-Zagats','iTunes-Amazon','Walmart-Amazon']:
    for MODE in ['MCAR','MAR','MNAR']:



        for frac in [0.1,0.2,0.3,0.4,0.5]:
            for rep in range(1,11):

                df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/test.csv')
                cols = []
                for col in df.columns:
                    col = col.replace('left_','').replace('right_','')
                    if col not in cols:
                        if col not in ['id','label']:
                            cols.append(col)


                if MODE == 'MCAR':
                    df = inject_MCAR(df,frac,cols)
                elif  MODE =='MAR':
                    df = inject_MAR(df,frac,cols)
                else:
                    df = inject_MNAR(df,frac,cols)




                model_path = 'VLDB_best_model_'+task+'.pth'
                dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
                embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
                df.to_csv(dataset_dir+'test_noisyWord'+str(rep)+str(1000*frac)+'.csv',index=False)
                test_file = 'test_noisyWord'+str(rep)+str(1000*frac)+'.csv'


                trainset = "/home/mmoslem3/DATA_VLDB/" + task + '/train.txt'
                validset = "/home/mmoslem3/DATA_VLDB/" + task+ '/valid.txt'
                testset = "/home/mmoslem3/DATA_VLDB/" + task +'/test_noisyWord'+str(rep)+str(1000*frac)+'.txt'


                
                dynamic_convert_csv_to_txt(trainset.replace('txt','csv'), trainset)
                dynamic_convert_csv_to_txt(validset.replace('txt','csv'), validset)
                dynamic_convert_csv_to_txt(testset.replace('txt','csv'), testset)


                data_dir = dataset_dir
                run_id = 0
                batch_size = 74
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
                                "trainset": "/home/mmoslem3/DATA_VLDB/" + task+ '/train.txt',
                                "validset": "/home/mmoslem3/DATA_VLDB/" + task+ '/valid.txt',
                                "testset": "/home/mmoslem3/DATA_VLDB/" + task+ '/test_noisyWord'+str(rep)+str(1000*frac)+'.txt',}, }



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
                # HP['logdir'] = os.path.join(logdir, task,str(fold))
                HP['task'] = task
                HP['alpha_aug'] = alpha_aug
                HP['fp16'] = fp16
                HP['lm'] = lm
                HP['lr'] = lr


                device = 'cuda:'+str(1) if torch.cuda.is_available() else 'cpu'

                model = DittoModel(device=device,lm=HP['lm'],alpha_aug=HP['alpha_aug'])

                model.to(device)
                ckpt_path = os.path.join('/home/mmoslem3/DITTO/checkpoints_vldb/Structured/'+task.replace('Fodors-Zagats','Fodors-Zagat'), 'model.pt')
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
                
                df = pd.concat([pd.DataFrame(all_probs, columns=['score']),pd.DataFrame(all_y, columns=['true'])], axis=1)            
                
                

                try: 
                    os.mkdir('/home/mmoslem3/RES_hierarch/DITTO_'+task+'_'+MODE)
                except:
                    pass
                df.to_csv('/home/mmoslem3/RES_hierarch/DITTO_'+task+'_'+MODE+ '/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
                print(MODE + '_'+str(frac2)+'_'+str(rep))