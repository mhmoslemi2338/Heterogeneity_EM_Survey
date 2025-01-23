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







import numpy as np
import pandas as pd
import time

import numpy as np
import pandas as pd
import time




date_hierarchy = {
    '1980s': {
        '1987': {
            '1987September': ['1987/September/23']
        }
    },
    '1990s': {
        '1990': {
            '1990January': ['1990/January/01']
        },
        '1994':[],
        '1997':[],
        '1995': {
            '1995June': ['1995/June/13']
        },
        '1999': {
            '1999May': ['1999/May/17'],
            '1999October': ['1999/October/19']
        }
    },
    '2000s': {
        '2000': {
            '2000August': ['2000/August/17'],
            '2000September': ['2000/September/26']
        },
        '2002': {
            '2002April': ['2002/April/02']
        },
        '2003': {
            '2003May': ['2003/May/20', '2003/May/27'],
            '2003August': ['2003/August/29'],
            '2003November': ['2003/November/11']
        },
        '2004': {
            '2004February': ['2004/February/02', '2004/February/10'],
            '2004July': ['2004/July/27'],
            '2004November': ['2004/November/16']
        },
        '2005': {
            '2005September': ['2005/September/20'],
            '2005October': ['2005/October/04'],
            '2005November': ['2005/November/07', '2005/November/22']
        },
        '2006': {
            '2006April': ['2006/April/04'],
            '2006June': ['2006/June/20'],
            '2006October':['2006/October/24'],
            '2006March':['2006/March/04']

        },
        '2007': {
             '2007September':['2007/September/18'],
            '2007February': ['2007/February/06'],
            '2007March': ['2007/March/16'],
            '2007July': ['2007/July/10'],
            '2007August': ['2007/August/14'],
            '2007November': ['2007/November/20']
        },
        '2008': {
            '2008March': ['2008/March/17', '2008/March/18'],
            '2008June': ['2008/June/23', '2008/June/24'],
            '2008July': ['2008/July/15', '2008/July/29'],
            '2008August': ['2008/August/01'],
            '2008October': ['2008/October/21', '2008/October/28'],
            '2008November': ['2008/November/04', '2008/November/11']
        },
        '2009': {
            '2009March': ['2009/March/30'],
            '2009April': ['2009/April/17'],
            '2009September':['2009/September/15'],
            '2009May': ['2009/May/05'],
            '2009August': ['2009/August/21'],
            '2009October': ['2009/October/02'],
            '2009November': ['2009/November/03', '2009/November/23'],
            '2009December': ['2009/December/21']
        }
    },










    '2010s': {
        '2010': {
             
            '2010June': ['2010/June/15', '2010/June/16', '2010/June/21'],
            '2010August': ['2010/August/03'],
            '2010October': ['2010/October/25'],
            '2010November': ['2010/November/24', '2010/November/26'],
            '2010December': ['2010/December/14', '2010/December/21','2010/December/03']
        },
        '2011': { 
             



            '2011January': ['2011/January/14'],
            '2011February': ['2011/February/08', '2011/February/14', '2011/February/22'],
            '2011March': ['2011/March/08', '2011/March/28', '2011/March/29'],
            '2011April': ['2011/April/18', '2011/April/26','2011/April/19'],
            '2011May': ['2011/May/23', '2011/May/25', '2011/May/27'],
            '2011June': ['2011/June/07'],
            '2011July': ['2011/July/14', '2011/July/22','2011/July/19'],
            '2011August': ['2011/August/23', '2011/August/26'],
            '2011September': ['2011/September/09', '2011/September/16', '2011/September/30'],
            '2011November': ['2011/November/15', '2011/November/21'],
            '2011December': ['2011/December/06', '2011/December/09', '2011/December/16', '2011/December/20', '2011/December/27','2011/December/12']
        },
        '2012': {
             




            '2012January': ['2012/January/20'],
            '2012February': ['2012/February/22'],
            '2012March': ['2012/March/20', '2012/March/23'],
            '2012April': ['2012/April/24','2012/April/17'],
            '2012May':['2012/May/11'],
            '2012June': ['2012/June/14', '2012/June/19', '2012/June/22'],
            '2012July': ['2012/July/02', '2012/July/31'],
            '2012August': ['2012/August/09', '2012/August/28'],
            '2012September':['2012/September/11'],
            '2012October': ['2012/October/09', '2012/October/30'],
            '2012November': ['2012/November/09', '2012/November/13'],
            '2012December': ['2012/December/04']
        },
        '2013': {
            '2013January': ['2013/January/04', '2013/January/18'],
            '2013February': ['2013/February/05', '2013/February/12','2013/February/27'],
            '2013March': ['2013/March/22', '2013/March/26'],
            '2013April': ['2013/April/02', '2013/April/05', '2013/April/16'],
            '2013May': ['2013/May/28'],
            '2013June': ['2013/June/11'],
            '2013July': ['2013/July/23'],
            '2013August': ['2013/August/06', '2013/August/13', '2013/August/20', '2013/August/27'],
            '2013September': ['2013/September/10', '2013/September/16', '2013/September/24', '2013/September/30'],
            '2013November': ['2013/November/08', '2013/November/11', '2013/November/25'],
            '2013December': ['2013/December/10', '2013/December/16']
        },
        '2014': {
            '2014February': ['2014/February/18', '2014/February/24','2014/February/11'],
            '2014April': ['2014/April/15'],
            '2014May': ['2014/May/06', '2014/May/20', '2014/May/27','2014/May/02'],
            '2014June': ['2014/June/02', '2014/June/04', '2014/June/20', '2014/June/23'],
            '2014July':['2014/July/01','2014/July/07'],
            '2014August': ['2014/August/19'],
            '2014September': ['2014/September/09'],
            '2014October': ['2014/October/14', '2014/October/21', '2014/October/27'],
            '2014November': ['2014/November/04', '2014/November/10', '2014/November/24'],
            '2014December': ['2014/December/23']








        },
        '2015': {
            '2015January': ['2015/January/09', '2015/January/13', '2015/January/21', '2015/January/30'],
            '2015February': ['2015/February/13', '2015/February/24'],
            '2015March': ['2015/March/03', '2015/March/10'],
            '2015April': ['2015/April/06', '2015/April/10', '2015/April/17', '2015/April/21'],
            '2015May': ['2015/May/12', '2015/May/15', '2015/May/18'],
            '2015June': ['2015/June/01', '2015/June/03', '2015/June/09', '2015/June/16', '2015/June/22', '2015/June/23', '2015/June/30'],
            '2015July': ['2015/July/01', '2015/July/03', '2015/July/10'],
            '2015August': ['2015/August/07', '2015/August/10', '2015/August/28', '2015/August/31'],
            '2015September': ['2015/September/04', '2015/September/11', '2015/September/18', '2015/September/21', '2015/September/25'],
            '2015October': ['2015/October/09']
        }
    }
}











genre_hierarchy = {
    "alternative rock": [
        "alternative rock",
        "alternative rock, indie & lo-fi",
        "alternative, music, rock, adult alternative"
    ],
    "country": [
        "country",
        "country, contemporary country",
        "country, country rock, pop, rock",
        "country, country rock, rock",
        "country, traditional country, cowboy",
        "country, contemporary country, honky tonk",
        "country, honky tonk, contemporary country",
        "country, honky tonk, pop, pop/rock, contemporary country, rock",
        "country, honky tonk, urban cowboy, contemporary country",
        "country, music, pop, teen pop, honky tonk, contemporary country",
        "country, music, rock, contemporary country, pop, pop/rock, honky tonk",
        "country, music, urban cowboy, contemporary country",
        "country, music, urban cowboy, honky tonk, contemporary country",
        "country, music, contemporary country",
        "country, music, contemporary country, honky tonk, traditional country",


                'country, music, honky tonk, contemporary country',
                'country, music, honky tonk, pop, pop/rock, contemporary country, rock',
                'country, music, honky tonk, urban cowboy, contemporary country'
                

    ],
    "dance & electronic": [
        "dance & electronic",
        "dance & electronic, dubstep",
        "dance & electronic, house",
        "dance & electronic, pop, house",
        "dance & electronic, pop, r&b",
        "dance & electronic, rap & hip-hop, house",
        "dance, music",
        "dance, music, electronic",
        "dance, music, electronic, classical, classical crossover, rock, house, electronica",
        "dance, music, electronic, house, rock, french pop",
        "dance, music, pop",
        "dance, music, r&b / soul, electronic",
        "dance, music, rock",
        "dance, music, rock, electronic",
        "dance, music, rock, house, electronic",
        "dance, music, rock, house, electronic, french pop",
        "dance, music, rock, pop, house, electronic, electronica, adult alternative",
        "dance, music, hip-hop / rap, alternative rap, hip-hop, r&b / soul, soul, electronic",
        "dance, music, hip-hop / rap, dirty south, rap, electronic",
        'dance,music,hip-hop / rap, alternative rap,hip-hop, r&b / soul, soul, electronic',
'dance,music,hip-hop / rap, dirty south, rap, electronic',
    ],
    "electronica": [
        "electronica, dance & electronic",
        "electronica, dance & electronic, dubstep",
        "electronica, dance & electronic, house",
        "electronic, music",
        "electronic, music, dance, rock, electronica",
'electronic,music,hip-hop / rap, rap',

    ],
    "folk": [
        "folk, rock"
    ],
    "gangsta & hardcore": [
        "gangsta & hardcore, rap & hip-hop"
    ],
    "gospel": [
        "gospel, christian"
    ],
    "holiday": [
        "holiday, christmas, miscellaneous"
    ],
    "international": [
        "international",
        "international, latin music",
        "international, latin music, latin hip-hop"
    ],
    "miscellaneous": [
        "miscellaneous",
        'house, music, dance, rock, electronic',

    ],
    "pop": [
        "pop",
        "r&b, pop",
        "pop, music, electronic, r&b / soul, pop/rock, dance",
        "pop, music, r&b / soul, contemporary r&b, hip-hop/rap, pop/rock, rock",
        "pop, music, r&b / soul, dance, teen pop, rock",
        "pop, music, r&b / soul, rock, dance, contemporary r&b",
        "pop, music, r&b / soul, teen pop, dance, rock",
        "pop, music, r&b / soul, soul, dance, rock, jazz, hip-hop / rap, electronic, hip-hop, pop/rock, adult alternative",
        "pop, music, rock",
        "pop, music, rock, pop/rock, teen pop, dance",
        "pop, music, rock, r&b / soul, contemporary r&b, dance",
        "pop, music, rock, r&b / soul, contemporary r&b, dance, electronic, hip-hop / rap, pop/rock",
        "pop, music, rock, singer/songwriter, contemporary singer/songwriter, adult alternative",
        "pop, music, r&b / soul, soul, dance, rock, jazz, hip-hop / rap, electronic, hip-hop, pop/rock, adult alternative",
        'pop, music',
'pop, music, r&b / soul,soul,dance,rock,jazz,hip-hop / rap,electronic,hip-hop, pop/rock, adult alternative',
'pop, music, rock, r&b / soul, contemporary r&b, dance,electronic,hip-hop / rap, pop/rock',
    ],
    "r&b": [
        "r&b"
    ],
    "rap & hip-hop": [
        "rap & hip-hop",
        "rap & hip-hop, southern rap, pop rap",
        "rap & hip-hop, west coast",
        "hip-hop/rap, music",
        "hip-hop/rap, music, dirty south",
        "hip-hop/rap, music, east coast rap, hardcore rap, rap",
        "hip-hop/rap, music, hardcore rap, east coast rap, rap",
        "hip-hop/rap, music, pop, dirty south",
        "hip-hop/rap, music, r&b / soul, contemporary r&b, dance, rap",
        "hip-hop/rap, music, rap",
        "hip-hop/rap, music, rap, dirty south",
        "hip-hop/rap, music, rap, east coast rap, hardcore rap",
        "hip-hop/rap, music, rap, electronic, world, dance",
        "hip-hop/rap, music, rap, hardcore rap, east coast rap, rock",
        "hip-hop/rap, music, rock, gangsta rap, west coast rap",
        "dance, music, hip-hop / rap, alternative rap, hip-hop, r&b / soul, soul, electronic",
        "dance, music, hip-hop / rap, dirty south, rap, electronic"
    ],
    "rock": [
        "rock",
        "rock, music, hard rock, alternative",
        "rock, music, metal, alternative, hard rock",
        "singer/songwriter, music, rock",
        'rock, music'
    ],
    "soundtracks": [
        "soundtracks",
        "soundtrack, music, soundtrack, classical, original score",
        "soundtrack, music, hip-hop / rap, west coast rap, gangsta rap, hardcore rap, rap, soundtrack",
        'soundtrack,music,hip-hop / rap, west coast rap, gangsta rap, hardcore rap, rap, soundtrack'

    ]
}


price_hierarchy = {
    "tier one": [
        '$ 0.69',
        '$ 0.89'
    ],
    "tier two": [
        "$ 0.99",'$ 1.29','$ 1.99'
    ],}


def make_mask(df,frac):
    L = len(COL)*len(df)
    L_disturb = int(np.ceil(L*frac))
    disturb_mask = np.zeros((len(df),len(COL)))
    total_elements = disturb_mask.size
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


def convert_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds


def disturb_func(col,value):
    if col =='Price':
        if value == 'Album Only':
            new_velue = 'other'
        else:
            parents = find_parents(price_hierarchy, value)
            if parents == -1: print(value)
            new_seed = int(time.time())
            np.random.seed(new_seed)
            np.random.shuffle(parents)
            new_velue = parents[0]
        

    elif col == 'Genre':
        parents = find_parents(genre_hierarchy,value.lower().replace(' , ',', '))
        if parents == -1: print(value)
        new_seed = int(time.time())
        np.random.seed(new_seed)
        np.random.shuffle(parents)
        new_velue = parents[0]


    elif col == 'Released':
        # print(value)



        if str(value) =='nan':
            new_velue = 'other'            

        elif '-' in value:
            value = value.split('-')
            
            if len(value[0]) ==1:
                value[0] = '0'+str(value[0])

            if int(value[2]) > 20:
                yy = '19'+str(value[2])
            else:
                yy = '20'+str(value[2])
                tr_dict = {'Jan':'January',
                        'Feb':'February',
                        'Mar':'March',
                        'Apr':'April',
                        'May':'May',
                        'Jun':'June',
                        'Jul':'July',
                        'Aug':'August',
                        'Sep':'September',
                        'Oct':'October',
                        'Nov':'November',
                        'Dec':'December'}

            value =  yy + '/'   + tr_dict[value[1]]+'/'+ value[0]
            parents = find_parents(date_hierarchy, value)

        elif ',' in value:
            value = value.replace(', ','').split(' ')
            
            if len(value[1]) ==1:
                value[1] = '0'+value[1]

                    
            value= value[2]+'/'+value[0]+'/'+value[1]
            parents = find_parents(date_hierarchy, value)

        else:
            parents = find_parents(date_hierarchy, value)
        if str(value) !='nan':
            # print(parents,value)
            new_seed = int(time.time())
            np.random.seed(new_seed)
            np.random.shuffle(parents)
            new_velue = parents[0]

    elif col =='Time':
        if str(value) =='nan':
            new_velue = 'other'

        else:

            data_in_seconds = [convert_to_seconds(length) for length in [value]]
            music_hierarchy = {
                "Short": {
                    "Very Short": [n for n in data_in_seconds if n < 60],
                    "Short high": [n for n in data_in_seconds if 60 <= n < 180]
                },
                "Moderate": {
                    "Moderate Low": [n for n in data_in_seconds if 180 <= n < 300],
                    "Moderate High": [n for n in data_in_seconds if 300 <= n < 480]
                },
                "Long": {
                    "Long Low": [n for n in data_in_seconds if 480 <= n < 600],
                    "Long High": [n for n in data_in_seconds if 600 <= n < 900]
                },
                "Very Long": {
                    "Very Long Low": [n for n in data_in_seconds if 900 <= n < 1200],
                    "Very Long High": [n for n in data_in_seconds if n >= 1200]
                }
            }
            parents = find_parents(music_hierarchy, data_in_seconds[0])
            if parents == -1:
                print(value,'time')
            new_seed = int(time.time())
            np.random.seed(new_seed)
            np.random.shuffle(parents)
            new_velue = parents[0]
        
    return new_velue




task = 'iTunes-Amazon'
COL = ['Genre', 'Price', 'Time','Released']


import os
import random
for frac in [0.1,0.2,0.3,0.4,0.5]:
    for rep in range(1,16):
        # df = pd.read_csv(f'data/{task}/test.csv')
        df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/test.csv')
        mask_left = make_mask(df,frac)
        mask_right = make_mask(df,frac)
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


        model_path = 'vldb_saved_model_'+task +'/'
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
        df.to_csv('/home/mmoslem3/RES_hierarch/HierGATitun/HG_score_'+task + '_'+frac2+'_'+str(rep)+'.csv', index=False)  
        # print("test_"+frac+'_'+str(rep)+".csv")
    # os.remove(data_dir+'/'+'train.txt')
    # os.remove(data_dir+'/'+'valid.txt')




