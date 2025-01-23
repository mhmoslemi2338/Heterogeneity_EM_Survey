import numpy as np
import pandas as pd
import os
import torch
import deepmatcher as dm



gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

for task in ['Fodors-Zagats','iTunes-Amazon','Walmart-Amazon']:
    for frac in [0.05,0.1,0.15,0.2,0.25]:



        for rep in range(1,4):


            model_path = 'lbl_noise'+task+str(100*frac+1)+'.pth'
            dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
            embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
            




            df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/train.csv')
            num_to_flip = int(len(df) * frac)
            indices_to_flip = np.random.choice(df.index, size=num_to_flip, replace=False)
            df.loc[indices_to_flip, 'label'] = 1 - df.loc[indices_to_flip, 'label']
            df.to_csv(dataset_dir+'train2.csv',index=False)
            


            df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/valid.csv')
            num_to_flip = int(len(df) * frac)
            indices_to_flip = np.random.choice(df.index, size=num_to_flip, replace=False)
            df.loc[indices_to_flip, 'label'] = 1 - df.loc[indices_to_flip, 'label']
            df.to_csv(dataset_dir+'vlaid2.csv',index=False)





            train_file = "train2.csv"
            valid_file = "valid2.csv"    
            test_file = "test.csv"
            try:
                os.remove(dataset_dir+'cacheddata.pth')
            except:
                pass


            datasets = dm.data.process(path=dataset_dir,train=train_file, validation=valid_file,test=test_file, embeddings_cache_path=embedding_dir)
            train, validation, test = datasets[0], datasets[1], datasets[2] if len(datasets)>=3 else None

            
            model = dm.MatchingModel()
            model.run_train(train, validation, best_save_path=model_path, epochs = 100)






            torch.cuda.empty_cache()                    
            pred_y = model.run_prediction(test)
            y_true = np.array(pd.read_csv(dataset_dir + test_file)['label'])
            y_score = list(pred_y['match_score'])



            df = pd.concat([pd.DataFrame(y_score), pd.DataFrame(y_true)], axis=1)
            frac2 = str(int(frac*100))
            if len(frac2) == 1: frac2 = frac2 +'0'

            try: 
                os.mkdir('/home/mmoslem3/RES_hierarch/DeepMatcher_'+task+'_'+'lblnoise')
            except:
                pass
            df.to_csv('/home/mmoslem3/RES_hierarch/DeepMatcher_'+task+'_'+'lblnoise'+ '/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
            print(task + '_'+str(frac2)+'_'+str(rep))