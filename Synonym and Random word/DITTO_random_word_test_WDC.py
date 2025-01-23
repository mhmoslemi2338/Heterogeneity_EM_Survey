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



gpu_no = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
NUM =-0

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






for task in ['WDC_folded']:
    for fold in [1,2,3,4,5]:
        data_dir = "/home/mmoslem3/DATA_synonym/DATA/" + task  + "/" + str(fold)






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







        # set seeds
        random.seed(run_id)
        np.random.seed(run_id)
        torch.manual_seed(run_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_id)


        # create the tag of the run
        run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, lm, da,dk, summarize, str(size), run_id)
        run_tag = run_tag.replace('/', '_')


        # load task configuration
        configs = json.load(open('config_random_WDC.json'))
        configs = {conf['name'] : conf for conf in configs}
        config = configs[task+'_'+str(fold)]


        trainset = config['trainset']
        validset = config['validset']
        testset = config['testset']

        dynamic_convert_csv_to_txt(trainset.replace('txt','csv'), trainset)
        dynamic_convert_csv_to_txt(validset.replace('txt','csv'), validset)
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
        HP['logdir'] = os.path.join(logdir, task,str(fold))
        HP['task'] = task
        HP['alpha_aug'] = alpha_aug
        HP['fp16'] = fp16
        HP['lm'] = lm
        HP['lr'] = lr

        # train(train_dataset,
        #         valid_dataset,
        #         test_dataset,
        #         run_tag, HP)


        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #----------------------------------------------------------------
        #----------------------------------------------------------------

        device = 'cuda:'+str(NUM) if torch.cuda.is_available() else 'cpu'

        model = DittoModel(device=device,lm=HP['lm'],alpha_aug=HP['alpha_aug'])

        model.to(device)
        directory = HP['logdir']
        ckpt_path = os.path.join(HP['logdir'],task, 'model.pt')
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model'])

        testset = config['testset']
        testset_base= "/home/mmoslem3/DATA_synonym/DATA/WDC_folded/" + str(fold) + '/'
            

        for frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
            torch.cuda.empty_cache()
            frac = str(int(frac*100))
            if len(frac) == 1: frac = frac +'0'
            for rep in range(1,11):

                
                df_original = pd.read_csv(testset_base+'/'+'test_00_'+str(rep)+'.csv')
                df = pd.read_csv(testset_base +'/'+"test_"+frac+'_'+str(rep)+".csv")
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


                
                # try: os.remove(testset_base+'/test_noisyWord.csv')
                # except: pass

                df.to_csv(testset_base +'/'+"noise_test_"+frac+'_'+str(rep)+".csv",index=False)

                test_path = "noise_test_"+frac+'_'+str(rep)+".csv"

                dynamic_convert_csv_to_txt(os.path.join(testset_base, test_path), os.path.join(testset_base, test_path.replace('csv','txt')))


                testset = testset_base +'/'+"noise_test_"+frac+'_'+str(rep)+".txt"


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
                        
                
                df = pd.concat([pd.DataFrame(all_probs, columns=['score']),pd.DataFrame(all_y, columns=['true'])], axis=1)            
                df.to_csv('/home/mmoslem3/RES_folded/DITTO_WDC_random_noise/DITTO_score_'+task.replace('folded','')+'_'+str(fold)+'_'+frac+'_'+str(rep)+'.csv', index=False)  






