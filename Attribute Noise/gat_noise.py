import numpy as np
import pandas as pd
import os
import torch
import random
from model.model import TranHGAT
from torch.utils import data
from model.dataset import Dataset
import csv


gpu_no = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no



CUDA = 0
N_epoch = 100



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




            model_path = 'VLDB_best_model_'+task.replace('Fodors-Zagats','Fodors-Zagat')+'.pth'
            dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
            embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
            df.to_csv(dataset_dir+'test_noisyWord_gat.csv',index=False)





            data_dir = dataset_dir

            train_path = "train.csv"
            valid_path = "valid.csv"
            test_path = "test.csv"

            for i in [valid_path, train_path]:
                if os.path.exists(os.path.join(data_dir, i.replace('csv','txt'))): continue
                dynamic_convert_csv_to_txt(os.path.join(data_dir, i), os.path.join(data_dir, i.replace('csv','txt')))



            i = "test_noisyWord_gat.csv"
            dynamic_convert_csv_to_txt(os.path.join(data_dir, i), os.path.join(data_dir, i.replace('csv','txt')))


            run_id = 0
            batch_size =32

            max_len = 256
            lr = 1e-5
            n_epochs = N_epoch
            finetuning = True
            save_model = True


            

            model_path = 'vldb_saved_model_' + task.replace('Fodors-Zagats','Fodors-Zagat')+'/'
            lm_path = None
            split = True
            lm = 'bert'





            args = {
                "batch_size": batch_size,
                "lr": lr,
                "n_epochs": n_epochs,
                "finetuning": finetuning,
                "save_model": save_model,
                "model_path": model_path,
                "lm_path": lm_path,
                "lm": lm}







            trainset = os.path.join(data_dir,train_path.replace('csv','txt'))
            validset = os.path.join(data_dir,valid_path.replace('csv','txt'))
            testset = os.path.join(data_dir,test_path.replace('csv','txt'))


            # load train/dev/test sets
            train_dataset = Dataset(trainset, ["0", "1"], lm=lm, lm_path=lm_path, max_len=max_len, split=split)
            valid_dataset = Dataset(validset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)
            test_dataset = Dataset(testset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)




            device = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
            model = TranHGAT( train_dataset.get_attr_num(), device, finetuning, lm=lm, lm_path=lm_path)
            model.load_state_dict(torch.load(model_path+'/'+'model.pt', map_location= device))
            model = model.to(device)
            model.eval()





            torch.cuda.empty_cache()
            frac2 = str(int(frac*100))
            if len(frac2) == 1: frac2 = frac2 +'0'



            test_path = 'test_noisyWord_gat.csv'







            

            testset = os.path.join(data_dir,test_path.replace('csv','txt'))
            test_dataset = Dataset(testset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)





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

            try: 
                os.mkdir('/home/mmoslem3/RES_hierarch/HierGAT_'+task+'_featurenoise')
            except:
                pass
            df.to_csv('/home/mmoslem3/RES_hierarch/HierGAT_'+task+'_featurenoise'+'/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
            print(str(frac2)+'_'+str(rep))