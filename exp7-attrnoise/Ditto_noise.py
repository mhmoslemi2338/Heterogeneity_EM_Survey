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







def make_mask(df,frac):
    L = len(cols)*len(df)
    L_disturb = int(np.ceil(L*frac))
    disturb_mask = np.zeros((len(df),len(cols)))
    total_elements = disturb_mask.size
    random_indices = np.random.choice(total_elements, L_disturb, replace=False)
    disturb_mask.flat[random_indices] = 1
    return disturb_mask 


import string
def inject_noise(value):
    if type(value) == str:
        num_typos = max(1, int(0.3 * len(value)))  # At least 1 typo for short strings
        value_list = list(value)  # Convert to list for mutability
        L = len(value_list)
        poses = np.random.choice(range(L), size=num_typos, replace=False)
        for pos in poses:
            random_char = random.choice(string.ascii_letters)  # Random replacement
            value_list[pos] = random_char  # Replace character
        value_list.append(random.choice(string.ascii_letters))
        return ''.join(value_list)

    if str(value) == 'nan': return np.nan
    return round(value + np.random.uniform(-value*0.2,value*0.2))








for task in ['Fodors-Zagats','iTunes-Amazon','Walmart-Amazon']:

    for frac in [0,0.1,0.2,0.3,0.4,0.5]:
        if frac ==0:
            R = range(1,2)
        else:
            R= range(1,11)
        for rep in R:

            df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/test.csv')
            cols = []
            for col in df.columns:
                col = col.replace('left_','').replace('right_','')
                if col not in cols:
                    if col not in ['id','label']:
                        cols.append(col)

            mask_left = make_mask(df,frac)
            mask_right = make_mask(df,frac)

            for i,col in enumerate(cols):
                df.loc[mask_left[:,i] == 1, 'left_'+col] = df.loc[mask_left[:,i] == 1, 'left_'+col].apply(lambda x: inject_noise(x))
                df.loc[mask_right[:,i] == 1, 'right_'+col] = df.loc[mask_right[:,i] == 1, 'right_'+col].apply(lambda x: inject_noise( x))



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
                os.mkdir('/home/mmoslem3/RES_hierarch/DITTO_'+task+'_featurenoise')
            except:
                pass
            df.to_csv('/home/mmoslem3/RES_hierarch/DITTO_'+task+'_featurenoise'+'/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
            print(str(frac2)+'_'+str(rep))