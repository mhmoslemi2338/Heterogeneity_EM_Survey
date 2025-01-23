import numpy as np
import pandas as pd
import os
import random
import torch
import deepmatcher as dm


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






gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


for task in ['Fodors-Zagats','iTunes-Amazon','Walmart-Amazon']:
    for MODE in ['MCAR','MAR','MNAR']:



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


                if MODE == 'MCAR':
                    df = inject_MCAR(df,frac,cols)
                elif  MODE =='MAR':
                    df = inject_MAR(df,frac,cols)
                else:
                    df = inject_MNAR(df,frac,cols)

                    




                model_path = 'VLDB_best_model_'+task+'.pth'
                dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
                embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
                df.to_csv(dataset_dir+'test_noisyWord2.csv',index=False)
                test_file = 'test_noisyWord2.csv'






                torch.cuda.empty_cache()                    
                try: os.remove(dataset_dir  +  'tmp2.pth')
                except: pass
                datasets = dm.data.process(path=dataset_dir,
                                                train='valid.csv',
                                                validation='valid.csv',
                                                test=test_file,
                                                embeddings_cache_path=embedding_dir,cache = 'tmp2.pth')
                train, validation, test = datasets[0], datasets[1], datasets[2] if len(datasets)>=3 else None



                model = dm.MatchingModel()
                model.load_state(model_path)

                pred_y = model.run_prediction(test)
                y_true = np.array(pd.read_csv(dataset_dir + test_file)['label'])
                y_score = list(pred_y['match_score'])



                df = pd.concat([pd.DataFrame(y_score), pd.DataFrame(y_true)], axis=1)
                frac2 = str(int(frac*100))
                if len(frac2) == 1: frac2 = frac2 +'0'
                try: 
                    os.mkdir('/home/mmoslem3/RES_hierarch/DeepMatcher_'+task+'_'+MODE)
                except:
                    pass
                df.to_csv('/home/mmoslem3/RES_hierarch/DeepMatcher_'+task+'_'+MODE+ '/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
                print(MODE + '_'+str(frac2)+'_'+str(rep))






