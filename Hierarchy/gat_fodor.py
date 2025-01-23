import numpy as np
import pandas as pd

import os
import json
import torch
import numpy as np
import random

from torch.utils import data

import pandas as pd
import csv


import random


import pandas as pd
import torch
from model.model import TranHGAT
from torch.utils import data
from model.dataset import Dataset
from train import initialize_and_train
import csv
import os

import random

import nltk
nltk.download('words')
from nltk.corpus import words
word_list = words.words()


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


#print(0)






city_hierarchy = {
    "united states": {
        "california": {
            "los angeles area": [
                "los angeles (la)", "west la", "hollywood", "w. hollywood", 'los angeles','la',
                "century", "chinatown", "studio", "westwood", "venice", 
                "los feliz", "encino", "sherman oaks", "santa monica", 
                "malibu", "beverly hills", "bel air", "manhattan beach", 
                "rancho park"
            ],
            "san francisco area": ["san francisco"],
            "other california": ["pasadena"]
        },
        "new york": {
            "new york city (nyc)": ["manhattan (new york)", "brooklyn", "queens"],
            "other new york": ["marietta"]
        },
        "georgia": {
            "atlanta": ["roswell"]
        },
        "nevada": {
            "las vegas": []
        },
        "other locations": ["st. hermosa beach",'lys velas']
    }
}

restaurant_hierarchy = {
    "american": {
        "styles": ["american (new)", "american (traditional)", "dive american",'american ( new )'],
        "barbecue": ["bbq"],
        "southern": ["cajun", "southern", "southern/soul", "southwestern"],
        "fast food & casual": [
            "hamburgers", "hot dogs", "fast food", "steak houses", 'chicken',
            "steakhouses", "health food", "vegetarian", "sandwiches",'delicatessen','delis'
        ],
        "coffee": ["coffee bar", "coffee shops", "coffee shops/diners", "coffeehouses"],
        "dining": ["diners", "cafeterias", "buffets"]
    },
    "asian": ["chinese", "japanese", "thai", "vietnamese", "indian", "indonesian"],
    "european": {
        "french": ["french", "french (classic)", "french (new)", "french bistro",'french ( new )','french ( classic )'],
        "others": [
            "greek", "greek and middle eastern", "italian", 
            "nuova cucina italian", "east european", 
            "polish", "russian", "scandinavian",'continental'
        ]
    },
    "latin american & caribbean": [
        "mexican", "tex-mex", "latin american", "mexican/latin american/spanish",
        "caribbean", "cuban", "tel caribbean"
    ],
    "mediterranean": ["mediterranean", "greek", "greek and middle eastern"],
    "seafood": ["seafood"],
    "eclectic/international": [
        "eclectic", "international", "pacific new wave", 
        "pacific rim", "californian", "old san francisco", 
        "only in las vegas", "ext 6108 international", 
        "or 212/632-5100 american"
    ],
    "desserts & specialty": ["desserts", "coffeehouses"],
    "pizza": ["pizza"]
}





# col = 'type'
# data = list(np.unique(list(np.unique(list(pd.read_csv(f'data/{task}/tableA.csv')[col])))+list(np.unique(list(pd.read_csv(f'data/{task}/tableB.csv')[col])))))
task = 'Fodors-Zagats'
COL = ['class','type','city']


def make_mask(df,frac):
    L = len(COL)*len(df)
    L_disturb = int(np.ceil(L*frac))
    disturb_mask = np.zeros((len(df),len(COL)))
    total_elements = disturb_mask.size
    import time

    # Use the current time to generate a new seed
    new_seed = int(time.time())
    np.random.seed(new_seed)
    random_indices = np.random.choice(total_elements, L_disturb, replace=False)
    disturb_mask.flat[random_indices] = 1
    return disturb_mask 


def find_parents(hierarchy, target, parents=None):
    if parents is None:
        parents = []

    for key, value in hierarchy.items():
        if key == target:  # If the target itself is a key
            return parents + [key]
        if isinstance(value, dict):  # If value is a nested dictionary
            result = find_parents(value, target, parents + [key])
            if result != -1:
                return result
        elif isinstance(value, list) and target in value:  # If value is a list containing the target
            return parents + [key]

    return -1  # Return -1 if the target is not found


def disturb_func(col,value):
    if col =='city':
        
        parents = find_parents(city_hierarchy, value.lower())
        if parents == -1: print(value)
        import time

        # Use the current time to generate a new seed
        new_seed = int(time.time())
        np.random.seed(new_seed)
        np.random.shuffle(parents)
        new_velue = parents[0]
        

    elif col == 'type':
        parents = find_parents(restaurant_hierarchy, value.lower())
        if parents == -1: print(value)
        np.random.shuffle(parents)
        new_velue = parents[0]

    elif col =='class':
        data = [int(value)]
        number_hierarchy = {
            "0-99": [n for n in data if 0 <= n <= 99],
            "100-199": [n for n in data if 100 <= n <= 199],
            "200-299": [n for n in data if 200 <= n <= 299],
            "300-399": [n for n in data if 300 <= n <= 399],
            "400-499": [n for n in data if 400 <= n <= 499],
            "500-599": [n for n in data if 500 <= n <= 599],
            "600-699": [n for n in data if 600 <= n <= 699],
            "700-799": [n for n in data if 700 <= n <= 799]
        }
        new_velue = next((key for key, values in number_hierarchy.items() if values), None)

    return new_velue



# /home/mmoslem3/DATA_VLDB/Fodors-Zagat/

import os
gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


import random
import torch
# import deepmatcher as dm
for frac in [0,0.1,0.2,0.3,0.4,0.5]:


    for rep in range(1,16):
        # df = pd.read_csv(f'data/{task}/test.csv')
        df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/test.csv')
        # df = pd.read_csv('tmp.csv')


        mask_left = make_mask(df,frac)
        mask_right = make_mask(df,frac)





        df['left_class'] = df['left_class'].astype(str)
        df['right_class'] = df['right_class'].astype(str)
        for i,col in enumerate(COL):

            

            
            df.loc[mask_left[:,i] == 1, 'left_'+col] = df.loc[mask_left[:,i] == 1, 'left_'+col].apply(lambda x: disturb_func(col, x))
            df.loc[mask_right[:,i] == 1, 'right_'+col] = df.loc[mask_right[:,i] == 1, 'right_'+col].apply(lambda x: disturb_func(col, x))



        model_path = 'VLDB_best_model_'+task+'.pth'
        dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
        embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
        df.to_csv(dataset_dir+'test_noisyWord.csv',index=False)





        data_dir = dataset_dir

        train_path = "valid.csv"
        valid_path = "valid.csv"
        test_path = "test.csv"

        for i in [valid_path, train_path]:
            if os.path.exists(os.path.join(data_dir, i.replace('csv','txt'))): continue
            dynamic_convert_csv_to_txt(os.path.join(data_dir, i), os.path.join(data_dir, i.replace('csv','txt')))



        i = "test_noisyWord.csv"
        dynamic_convert_csv_to_txt(os.path.join(data_dir, i), os.path.join(data_dir, i.replace('csv','txt')))


        run_id = 0
        batch_size =32

        max_len = 256
        lr = 1e-5
        n_epochs = N_epoch
        finetuning = True
        save_model = True


        model_path = 'vldb_saved_model_Fodors-Zagat/'
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
        frac2 = str(int(frac2*100))
        if len(frac2) == 1: frac2 = frac2 +'0'



        test_path = 'test_noisyWord.csv'







        

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
        df.to_csv('/home/mmoslem3/RES_hierarch/HierGATFodors-Zagats/HG_score_'+task + '_'+frac2+'_'+str(rep)+'.csv', index=False)  
        print("test_"+frac+'_'+str(rep)+".csv")
    # os.remove(data_dir+'/'+'train.txt')
    # os.remove(data_dir+'/'+'valid.txt')




