import deepmatcher as dm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import torch
import deepmatcher as dm
gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


#
#from nltk.corpus import words
import nltk
nltk.download('words')
from nltk.corpus import words
word_list = words.words()

for task in ['WDC_folded']:
    # for fold in [1,2,3,4,5]:
    for fold in [1]:
        model_path = 'WDC_foldedbest_model_fold'+str(fold)+'.pth'
        model_path = 'WDC_tmp.pth'
        dataset_dir = "/home/mmoslem3/DATA_synonym/DATA/" + task  + "/" + str(fold) + '/'
        embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
       # try: os.remove(dataset_dir +str(fold)+ 'cacheddata.pth')
       # except: pass




        # for frac in [0,0.1, 0.2, 0.3, 0.4,0.5]:
        for frac in [0,0.1]:
            
            frac = str(int(frac*100))
            if len(frac) == 1: frac = frac +'0'
            # for rep in range(1,11):
            for rep in range(1,3):
                

                # if os.path.exists('/home/mmoslem3/RES_folded/DeepMatcher_WDC_random_noise/DM_score_'+task.replace('folded','')+'_'+str(fold)+'_'+frac+'_'+str(rep)+'.csv'): continue
                if os.path.exists('/home/mmoslem3/RES_folded/tmp/DM_score_'+task.replace('folded','')+'_'+str(fold)+'_'+frac+'_'+str(rep)+'.csv'): continue
                df_original = pd.read_csv(dataset_dir+'test_00_'+str(rep)+'.csv')
                df = pd.read_csv(dataset_dir+'test_'+str(frac)+'_'+str(rep)+'.csv')
                


                # for col in ['left_name','right_name','left_description','right_description']:


                for col in ['left_brand','left_title','left_description','left_price','left_priceCurrency','right_brand','right_title','right_description','right_price','right_priceCurrency']:


                    for i in range(df.shape[0]):
                        
                        
                        str1 = df[col][i]
                        str2 = df_original[col][i]
                        if str(df[col][i]) == 'nan': str1 = 'nan'
                        if str(df_original[col][i]) == 'nan': str2 = 'nan'  
                        str1 = str1.replace('\xa0',' ').replace('light-emitting diode','light-emitting-diode').replace('"Digital Theater System"','"Digital-Theater-System"')
                        str2 = str2.replace('\xa0',' ').replace('light-emitting diode','light-emitting-diode').replace('"Digital Theater System"','"Digital-Theater-System"')
                        noisy = str1.split(' ')
                        orig = str2.split(' ')
                        for idx,row in enumerate(orig):
                            try:
                                if row!= noisy[idx]:
                                    new_word = random.choice(word_list).lower()
                                    noisy[idx] = new_word
                            except:
                                print(idx)
                        noisy = " ".join(noisy)
                        df.loc[i, col] = noisy


                df.to_csv(dataset_dir+'test_noisyWord2.csv',index=False)
                test_file = 'test_noisyWord2.csv'



                torch.cuda.empty_cache()                    
                try: os.remove(dataset_dir +str(fold) +  'tmp2.pth')
                except: pass
                datasets = dm.data.process(path=dataset_dir,
                                                train='tmp.csv',
                                                validation='tmp.csv',
                                                test=test_file,
                                                embeddings_cache_path=embedding_dir,cache = 'tmp2.pth')
                train, validation, test = datasets[0], datasets[1], datasets[2] if len(datasets)>=3 else None



                model = dm.MatchingModel()
                model.load_state(model_path)

                pred_y = model.run_prediction(test)
                y_true = np.array(pd.read_csv(dataset_dir + test_file)['label'])
                y_score = list(pred_y['match_score'])




                df = pd.concat([pd.DataFrame(y_score), pd.DataFrame(y_true)], axis=1)
                # df.to_csv('/home/mmoslem3/RES_folded/DeepMatcher_WDC_random_noise/DM_score_'+task.replace('folded','')+'_'+str(fold)+'_'+frac+'_'+str(rep)+'.csv', index=False)  
                df.to_csv('/home/mmoslem3/RES_folded/tmp/DM_score_'+task.replace('folded','')+'_'+str(fold)+'_'+frac+'_'+str(rep)+'.csv', index=False)  
                print('test_'+str(frac)+'_'+str(rep)+'.csv')







