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

                    






                model_path = 'VLDB_best_model_'+task.replace('Fodors-Zagats','Fodors-Zagat')+'.pth'
                dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
                embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
                df.to_csv(dataset_dir+'test_noisyWord_gat.csv',index=False)





                data_dir = dataset_dir

                train_path = "valid.csv"
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
                    os.mkdir('/home/mmoslem3/RES_hierarch/HierGAT_'+task+'_'+MODE)
                except:
                    pass
                df.to_csv('/home/mmoslem3/RES_hierarch/HierGAT_'+task+'_'+MODE+ '/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
                print(MODE + '_'+str(frac2)+'_'+str(rep))