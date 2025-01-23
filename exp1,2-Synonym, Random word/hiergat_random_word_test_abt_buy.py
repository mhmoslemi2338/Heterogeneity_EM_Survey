
import pandas as pd
import torch
from model.model import TranHGAT
from torch.utils import data
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from model.dataset import Dataset
from train import initialize_and_train
import csv
import os
import warnings

import contextlib
import io
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

for task in ['abt_buy_folded']:
    for fold in [1,2,3,4,5]:
        data_dir = "/home/mmoslem3/DATA_synonym/DATA/" + task  + "/" + str(fold) 
        model_path = 'best_model_fold'+str(fold)+'.pth'

        train_path = "train.csv"
        valid_path = "valid.csv"
        test_path = "test_00_1.csv"

        for i in [valid_path, train_path]:
            if os.path.exists(os.path.join(data_dir, i.replace('csv','txt'))): continue
            dynamic_convert_csv_to_txt(os.path.join(data_dir, i), os.path.join(data_dir, i.replace('csv','txt')))



        i = "test_00_1.csv"
#        if os.path.exists(os.path.join(data_dir, i.replace('csv','txt'))): continue
        dynamic_convert_csv_to_txt(os.path.join(data_dir, i), os.path.join(data_dir, i.replace('csv','txt')))




        run_id = 0
        batch_size =32

        max_len = 256
        lr = 1e-5
        n_epochs = N_epoch
        finetuning = True
        save_model = True


        model_path = "saved_model_random_word_"+task+'_' + str(fold)+"/"
        lm_path = None
        split = True
        lm = 'bert'






        trainset = os.path.join(data_dir,train_path.replace('csv','txt'))
        validset = os.path.join(data_dir,valid_path.replace('csv','txt'))
        testset = os.path.join(data_dir,test_path.replace('csv','txt'))


        # load train/dev/test sets
        train_dataset = Dataset(trainset, ["0", "1"], lm=lm, lm_path=lm_path, max_len=max_len, split=split)
        valid_dataset = Dataset(validset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)
        test_dataset = Dataset(testset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)



        args = {
            "batch_size": batch_size,
            "lr": lr,
            "n_epochs": n_epochs,
            "finetuning": finetuning,
            "save_model": save_model,
            "model_path": model_path,
            "lm_path": lm_path,
            "lm": lm}




 #       print(1)
        initialize_and_train(train_dataset, valid_dataset, test_dataset, train_dataset.get_attr_num(), args, '1',CUDA= CUDA, EARLY = 10)


        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #----------------------------------------------------------------





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




        for frac in [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

            torch.cuda.empty_cache()
            frac = str(int(frac*100))
            if len(frac) == 1: frac = frac +'0'
            for rep in range(1,11):
                if os.path.exists('/home/mmoslem3/RES_folded/HierGAT_abt_buy_random_noise/HG_score_'+task.replace('folded','')+'_'+str(fold)+'_'+frac+'_'+str(rep)+'.csv'): continue


                test_path = "test_"+frac+'_'+str(rep)+".csv"
                df_original = pd.read_csv(data_dir+'/'+'test_00_'+str(rep)+'.csv')
                df = pd.read_csv(data_dir +'/'+test_path)
                for col in ['left_name','right_name','left_description','right_description']:
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
                            if row!= noisy[idx]:
                                new_word = random.choice(word_list).lower()
                                noisy[idx] = new_word
                        noisy = " ".join(noisy)
                        df.loc[i, col] = noisy


                df.to_csv(data_dir+'/test_noisyWord.csv',index=False)
                test_path = 'test_noisyWord.csv'

        
        
                dynamic_convert_csv_to_txt(os.path.join(data_dir, test_path), os.path.join(data_dir, test_path.replace('csv','txt')))





                

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
                df.to_csv('/home/mmoslem3/RES_folded/HierGAT_abt_buy_random_noise/HG_score_'+task.replace('folded','')+'_'+str(fold)+'_'+frac+'_'+str(rep)+'.csv', index=False)  
                print("test_"+frac+'_'+str(rep)+".csv")
    os.remove(data_dir+'/'+'train.txt')
    os.remove(data_dir+'/'+'valid.txt')




