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
import nltk
nltk.download('words')
from nltk.corpus import words
word_list = words.words()
import random


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import pandas as pd
# from src.config import Config, create_experiment_folder
# from src.logging_customized import setup_logging
from src.data_loader import load_data, DataType
from src.data_representation import DeepMatcherProcessor
# from src.evaluation import Evaluation
# from src.model import save_model
# from src.optimizer import build_optimizer
# from src.training import train
import torch
from tqdm import tqdm


import random


import numpy as np
import pandas as pd
import torch
# import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_transformers import BertTokenizer
from pytorch_transformers.modeling_bert import BertForSequenceClassification
# from sklearn.metrics import roc_auc_score, roc_curve
from src.data_representation import DeepMatcherProcessor
from src.data_loader import load_data, DataType
import os
from src.config import Config

# gpu_no = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
# NUM =0

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
        df.to_csv(dataset_dir+'test_noisyWord.csv',index=False)



        data_dir = dataset_dir
        for isx in ['test_noisyWord.csv']:
            df = pd.read_csv(data_dir + '/'+ isx )
            df['combined_left'] = df.filter(like='left').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
            df['combined_right'] = df.filter(like='right').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
            df = df[['id' , 'combined_left','combined_right','label']]
            df = df.rename(columns={'id':'idx','combined_left': 'text_left', 'combined_right': 'text_right','label':'label'})
            df.to_csv(os.path.join(data_dir, isx.replace('csv','tsv')),sep='\t', index=False)






        CUDA = 2
        EPOCH = 100






    
        num_epochs = EPOCH
        model_type = 'bert' # bert
        model_name_or_path = 'bert-base-uncased' # "pre_trained_model/bert-base-uncased
        train_batch_size = 32
        eval_batch_size = 16
        max_seq_length = 256
        model_output_dir = 'vldb_MODEL_Fodors-Zagat'
        weight_decay = 0
        max_grad_norm = 1
        warmup_steps = 0 
        adam_eps = 1e-8
        learning_rate = 2e-5
        save_model_after_epoch = False
        do_lower_case = True





        config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(model_name_or_path)


        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenizer = tokenizer.from_pretrained(model_output_dir + '/' + model_output_dir, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config = config)
        model = model.from_pretrained(model_output_dir + '/' + model_output_dir)
        model.to(device)


        torch.cuda.empty_cache()






        processor = DeepMatcherProcessor()
        test_batch_size = 32 
        test_examples = processor.get_test_examples_fold(data_dir,frac, rep-1)
        test_data_loader = load_data(test_examples,
                                        processor.get_labels(),
                                        tokenizer,
                                        max_seq_length,
                                        test_batch_size,
                                        DataType.TEST,model_type)




        labels = None
        all_probs = []
        all_y = [] 
        for batch in tqdm(test_data_loader, desc="Test"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0].to(device),
                            'attention_mask': batch[1].to(device),
                            'token_type_ids': batch[2].to(device),
                            'labels': batch[3].to(device)}

                outputs = model(**inputs)
                _, logits = outputs[:2]

                labels = inputs['labels'].detach().cpu().numpy()
                probs = logits.softmax(dim=1)[:, 1]

                all_probs += probs.cpu().numpy().tolist()
                all_y += labels.tolist()


        y_true = all_y
        y_score = all_probs
        df = pd.DataFrame({'prob': y_score, 'label': y_true})
        os.remove(os.path.join(data_dir, isx.replace('csv','tsv')))
        frac2 = str(int(frac*100))
        if len(frac2) == 1: frac2 = frac2 +'0'

        df.to_csv('/home/mmoslem3/RES_hierarch/EMTFodors-Zagats/EM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  

