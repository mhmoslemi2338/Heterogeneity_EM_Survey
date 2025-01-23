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

gpu_no = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
NUM =0

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
        df.to_csv(dataset_dir+'test_noisyWord'+str(rep)+str(1000*frac)+'.csv',index=False)



        test_file = 'test_noisyWord'+str(rep)+str(1000*frac)+'.csv'




        data_dir = dataset_dir
        run_id = 0
        batch_size = 64
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




        # # set seeds
        # random.seed(run_id)
        # np.random.seed(run_id)
        # torch.manual_seed(run_id)   
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(run_id)


        # # create the tag of the run
        # run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, lm, da,dk, summarize, str(size), run_id)
        # run_tag = run_tag.replace('/', '_')


        # load task configuration
        configs = json.load(open('config_hier.json'))
        configs = {conf['name'] : conf for conf in configs}
        config = configs[task]


        trainset = config['trainset']
        validset = config['validset']
        testset = config['testset']


        testset = testset.replace('test_noisyWord2.txt','test_noisyWord'+str(rep)+str(1000*frac)+'.txt')
        
        

        # dynamic_convert_csv_to_txt(trainset.replace('txt','csv'), trainset)
        # dynamic_convert_csv_to_txt(validset.replace('txt','csv'), validset)
        dynamic_convert_csv_to_txt(testset.replace('txt','csv'), testset)




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


        device = 'cuda:'+str(0) if torch.cuda.is_available() else 'cpu'

        model = DittoModel(device=device,lm=HP['lm'],alpha_aug=HP['alpha_aug'])

        model.to(device)
        # directory = HP['logdir']
        ckpt_path = os.path.join('/home/mmoslem3/DITTO/checkpoints_vldb/Structured/Fodors-Zagat/', 'model.pt')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])

        testset = config['testset']
        testset = testset.replace('test_noisyWord2.txt','test_noisyWord'+str(rep)+str(1000*frac)+'.txt')
            

        torch.cuda.empty_cache()
        frac2 = str(int(frac*100))
        if len(frac2) == 1: frac2 = frac2 +'0'




        # testset = "/home/mmoslem3/DATA_VLDB/Fodors-Zagats/test_noisyWord2.txt"


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
        df.to_csv('/home/mmoslem3/RES_hierarch/DITTOFodors-Zagats/DITTO_score_'+task +'_'+frac2+'_'+str(rep)+'.csv', index=False)  








