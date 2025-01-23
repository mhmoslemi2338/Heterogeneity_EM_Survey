import random
import numpy as np
import pandas as pd
import time
import json
import torch
from model.model import TranHGAT
from torch.utils import data
from model.dataset import Dataset
from train import initialize_and_train
import csv
import os

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





for task in ['Fodors-Zagats','iTunes-Amazon','Walmart-Amazon']:
    for frac in [0.05,0.1,0.15,0.2,0.25]:


        if 'Walmart' in task: R = [1,2]
        else: R=[1,2,3,4,5]
        for rep in R:


            model_path = 'lbl_noise'+task+str(100*frac+1)+'.pth'
            dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
            embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
            data_dir = dataset_dir
            


            df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/train.csv')
            num_to_flip = int(len(df) * frac)
            indices_to_flip = np.random.choice(df.index, size=num_to_flip, replace=False)
            df.loc[indices_to_flip, 'label'] = 1 - df.loc[indices_to_flip, 'label']
            df.to_csv(dataset_dir+'train2.csv',index=False)
            


            df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/valid.csv')
            num_to_flip = int(len(df) * frac)
            indices_to_flip = np.random.choice(df.index, size=num_to_flip, replace=False)
            df.loc[indices_to_flip, 'label'] = 1 - df.loc[indices_to_flip, 'label']
            df.to_csv(dataset_dir+'valid2.csv',index=False)







            trainset = "/home/mmoslem3/DATA_VLDB/" + task + '/train2.txt'
            validset = "/home/mmoslem3/DATA_VLDB/" + task+ '/valid2.txt'
            testset = "/home/mmoslem3/DATA_VLDB/" + task +'/test.txt'


            
            dynamic_convert_csv_to_txt(trainset.replace('txt','csv'), trainset)
            dynamic_convert_csv_to_txt(validset.replace('txt','csv'), validset)
            dynamic_convert_csv_to_txt(testset.replace('txt','csv'), testset)




            run_id = 0
            batch_size = 16
            max_len = 1024
            lr = 1e-5
            n_epochs = 30
            finetuning = True
            save_model = True
            model_path = "hierTrain" +task+str(frac*100+1)+"/"
            lm_path = None
            split = True
            lm = 'bert'



            train_dataset = Dataset(trainset, ["0", "1"], lm=lm, lm_path=lm_path, max_len=max_len, split=split)
            valid_dataset = Dataset(validset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)
            test_dataset = Dataset(testset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)


            args = {
                "batch_size": batch_size,
                "lr": lr,
                "n_epochs": n_epochs,
                "finetuning": finetuning,
                "save_model": save_model,
                "model_path": model_path,
                "lm_path": lm_path,
                "lm": lm}
            
            initialize_and_train(train_dataset, valid_dataset, test_dataset, train_dataset.get_attr_num(), args, '1')





            device = 'cuda:'+str(NUM) if torch.cuda.is_available() else 'cpu'
            model = TranHGAT( train_dataset.get_attr_num(), device, finetuning, lm=lm, lm_path=lm_path)
            model.load_state_dict(torch.load(model_path+'/'+'model.pt', map_location= device))
            model = model.to(device)
            model.eval()



            torch.cuda.empty_cache()
            frac2 = str(int(frac2*100))
            if len(frac2) == 1: frac2 = frac2 +'0'



            all_probs = []
            all_y =[]

            test_iter = data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False, num_workers=0, collate_fn=Dataset.pad)
            with torch.no_grad():
                for i, batch in enumerate(test_iter):
                    _, x, y, _, masks = batch
                    logits, y1, y_hat = model(x.to(device), y.to(device), masks.to(device))
                    logits = logits.view(-1, logits.shape[-1])
                    probs = logits.softmax(dim=1)[:, 1]
                    all_probs += probs.cpu().numpy().tolist()
                    all_y += y1.cpu().numpy().tolist()
                    


            y_true = all_y
            y_score = all_probs
            df = pd.concat([pd.DataFrame(y_true, columns=['y_true']), pd.DataFrame(y_score, columns=['y_score'])], axis=1)

            if frac2 =='50': frac2 ='05'
            try:
                os.mkdir('/home/mmoslem3/RES_hierarch/HierGAT'+task+'_'+'lblnoise')
            except:
                pass
            df.to_csv('/home/mmoslem3/RES_hierarch/HierGAT'+task+'_'+'lblnoise'+'/HG_score_'+task + '_'+frac2+'_'+str(rep)+'.csv', index=False)  
            print("test_"+frac2+'_'+str(rep)+".csv")



