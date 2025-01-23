import numpy as np
import pandas as pd
import os





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
        "styles": ["american (new)", "american (traditional)", "dive american",'american ( new )','american ( traditional )','or 212/632 -5100 american'],
        "barbecue": ["bbq"],
        "southern": ["cajun", "southern", "southern/soul", "southwestern"],
        "fast food & casual": [
            "hamburgers", "hot dogs", "fast food", "steak houses", 'chicken','pizza',
            "steakhouses", "health food", "vegetarian", "sandwiches",'delicatessen','delis'
        ],
        "coffee": ["coffee bar", "coffee shops", "coffee shops/diners", "coffeehouses"],
        "dining": ["diners", "cafeterias", "buffets"]
    },
    "asian": ["chinese", "japanese", "thai", "vietnamese", "indian", "indonesian"],
    "european": {
        "french": ["french", "french (classic)", "french (new)", "french bistro",'french ( new )','french ( classic )'],
        "others": [
            "greek", "greek and middle eastern", "italian", 'spanish',
            "nuova cucina italian", "east european", 
            "polish", "russian", "scandinavian",'continental'
        ]
    },
    "latin american & caribbean": [
        "mexican", "tex-mex", "latin american", "mexican/latin american/spanish",
        "caribbean", "cuban", "tel caribbean",'seafood'
    ],
    "mediterranean": ["mediterranean", "greek", "greek and middle eastern"],
    "eclectic/international": [
        "eclectic", "international", "pacific new wave", 
        "pacific rim", "californian", "old san francisco", 
        "only in las vegas", "ext 6108 international", 
        "or 212/632-5100 american"
    ],
    "desserts & specialty": ["desserts", "coffeehouses"],
}




# col = 'type'
# data = list(np.unique(list(np.unique(list(pd.read_csv(f'data/{task}/tableA.csv')[col])))+list(np.unique(list(pd.read_csv(f'data/{task}/tableB.csv')[col])))))
task = 'Fodors-Zagat'
COL = ['class','type','city']


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


def disturb_func(col,value):
    if col =='city':
        
        parents = find_parents(city_hierarchy, value.lower())
        if parents == -1: print(value)

        np.random.shuffle(parents)
        new_velue = parents[0]
        

    elif col == 'type':
        if str(value) =='nan':
            new_velue = 'other'
        else:

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

# task = 'Walmart-Amazon'
gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

import os
import random
import torch
import deepmatcher as dm
for frac in [0.3,0.4,0.5,0.7,0.9]:



    for rep in range(3):


        model_path = 'hier'+task+str(100*frac)+'.pth'
        dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
        dataset_dir = '/Users/mohammad/Desktop/EM_Heterogeneity/data/Fodors-Zagat/'
        
        # embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
        




        # df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/train.csv')
        df = pd.read_csv("/Users/mohammad/Desktop/EM_Heterogeneity/data/Fodors-Zagat/train.csv")
        
        mask_left = make_mask(df,frac)
        mask_right = make_mask(df,frac)
        df['left_class'] = df['left_class'].astype(str)
        df['right_class'] = df['right_class'].astype(str)
        for i,col in enumerate(COL):
            df.loc[mask_left[:,i] == 1, 'left_'+col] = df.loc[mask_left[:,i] == 1, 'left_'+col].apply(lambda x: disturb_func(col, x))
            df.loc[mask_right[:,i] == 1, 'right_'+col] = df.loc[mask_right[:,i] == 1, 'right_'+col].apply(lambda x: disturb_func(col, x))
        df.to_csv(dataset_dir+'train_hier_deep.csv',index=False)
        


        # df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/valid.csv')
        df = pd.read_csv("/Users/mohammad/Desktop/EM_Heterogeneity/data/Fodors-Zagat/valid.csv")
        
        mask_left = make_mask(df,frac)
        mask_right = make_mask(df,frac)
        df['left_class'] = df['left_class'].astype(str)
        df['right_class'] = df['right_class'].astype(str)
        for i,col in enumerate(COL):
            df.loc[mask_left[:,i] == 1, 'left_'+col] = df.loc[mask_left[:,i] == 1, 'left_'+col].apply(lambda x: disturb_func(col, x))
            df.loc[mask_right[:,i] == 1, 'right_'+col] = df.loc[mask_right[:,i] == 1, 'right_'+col].apply(lambda x: disturb_func(col, x))
        df.to_csv(dataset_dir+'valid_hier_deep.csv',index=False)





        train_file = "train_hier_deep.csv"
        valid_file = "valid_hier_deep.csv"    
        test_file = "test.csv"
        try:
            os.remove(dataset_dir+'cacheddata.pth')
        except:
            pass

        embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 

        datasets = dm.data.process(path=dataset_dir,train=train_file, validation=valid_file,test=test_file, embeddings_cache_path=embedding_dir)
        train, validation, test = datasets[0], datasets[1], datasets[2]
        
        model = dm.MatchingModel()
        # model.run_train(train,validation,epochs=100,batch_size=32, pos_weight=1.5,best_save_path=model_path)
        model.run_train(train, validation, best_save_path=model_path, epochs = 100)









        model = dm.MatchingModel()
        model.load_state(model_path)



        pred_y = model.run_prediction(test)
        y_true = np.array(pd.read_csv(dataset_dir + test_file)['label'])
        y_score = list(pred_y['match_score'])



        df = pd.concat([pd.DataFrame(y_score), pd.DataFrame(y_true)], axis=1)
        frac2 = str(int(frac*100))
        if len(frac2) == 1: frac2 = frac2 +'0'

        try: 
            os.mkdir('/home/mmoslem3/RES_hierarch2/DeepMatcher_'+task+'_'+'hier')
        except:
            pass
        df.to_csv('/home/mmoslem3/RES_hierarch2/DeepMatcher_'+task+'_'+'hier'+ '/DM_score_'+task+'_'+frac2+'_'+str(rep+1)+'.csv', index=False)  
        print(task + '_'+str(frac2)+'_'+str(rep+1))





