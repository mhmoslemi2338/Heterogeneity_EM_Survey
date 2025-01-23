import numpy as np
import pandas as pd
import os
import random
import torch
import deepmatcher as dm




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




gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


for task in ['Fodors-Zagats','iTunes-Amazon','Walmart-Amazon']:



    for frac in [0, 0.1,0.2,0.3,0.4,0.5]:
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



            if 'Walmart' in task:
                model_path =  'new_walmart_jdiq.pth'
            else:
                model_path = 'VLDB_best_model_'+task+'.pth'
            dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
            embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
            df.to_csv(dataset_dir+'test_noisy.csv',index=False)
            test_file = 'test_noisy.csv'






            torch.cuda.empty_cache()                    
            try: os.remove(dataset_dir  +  'tmp.pth')
            except: pass
            datasets = dm.data.process(path=dataset_dir,
                                            train='train.csv',
                                            validation='valid.csv',
                                            test=test_file,
                                            embeddings_cache_path=embedding_dir,cache = 'tmp.pth')
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
                os.mkdir('/home/mmoslem3/RES_hierarch/DeepMatcher_'+task+'_featurenoise')
            except:
                pass
            df.to_csv('/home/mmoslem3/RES_hierarch/DeepMatcher_'+task+'_featurenoise'+ '/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
            print(str(frac2)+'_'+str(rep))






