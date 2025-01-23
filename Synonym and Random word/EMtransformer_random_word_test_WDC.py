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

import nltk
nltk.download('words')
from nltk.corpus import words
word_list = words.words()
import random





CUDA = 2
EPOCH = 100

for task in ['WDC_folded']:

    for fold in [1,2,3,4,5]:
        data_dir = "/home/mmoslem3/DATA_synonym/DATA/" + task  + "/" + str(fold) + '/'




    
        num_epochs = EPOCH
        model_type = 'bert' # bert
        model_name_or_path = 'bert-base-uncased' # "pre_trained_model/bert-base-uncased
        train_batch_size = 32
        eval_batch_size = 16
        max_seq_length = 256
        model_output_dir = 'MODEL_'+task + str(fold)+"/"
        weight_decay = 0
        max_grad_norm = 1
        warmup_steps = 0 
        adam_eps = 1e-8
        learning_rate = 2e-5
        save_model_after_epoch = False
        do_lower_case = True



        train_path = "train.csv"
        valid_path = "valid.csv"
        test_path = "tmp.csv"




        for isx in [train_path, valid_path]:
            df = pd.read_csv(data_dir + '/'+ isx )
            df['combined_left'] = df.filter(like='left').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
            df['combined_right'] = df.filter(like='right').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
            df = df[['id' , 'combined_left','combined_right','label']]
            df = df.rename(columns={'id':'idx','combined_left': 'text_left', 'combined_right': 'text_right','label':'label'})
            df.to_csv(os.path.join(data_dir, isx.replace('csv','tsv')),sep='\t', index=False)



        # device = torch.device("cuda:"+str(CUDA) if torch.cuda.is_available() else "cpu")
        # processor = DeepMatcherProcessor()
        # label_list = processor.get_labels()
        # config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[model_type]
        # config = config_class.from_pretrained(model_name_or_path)
        # tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        # model = model_class.from_pretrained(model_name_or_path, config=config)
        # model.to(device)







        # train_examples = processor.get_train_examples(data_dir) 
        # training_data_loader = load_data(train_examples,label_list,tokenizer,max_seq_length,train_batch_size,DataType.TRAINING, model_type)


        # num_train_steps = len(training_data_loader) * num_epochs
        # optimizer, scheduler = build_optimizer(model,num_train_steps,learning_rate,adam_eps,warmup_steps,weight_decay)
        # eval_examples = processor.get_dev_examples(data_dir)
        # evaluation_data_loader = load_data(eval_examples,label_list,tokenizer,max_seq_length,eval_batch_size,DataType.EVALUATION, model_type)


        # evaluation = Evaluation(evaluation_data_loader, model_output_dir, model_output_dir, len(label_list), model_type)




        # train(device,
        #         training_data_loader,
        #         model,
        #         optimizer,
        #         scheduler,
        #         evaluation,
        #         num_epochs,
        #         max_grad_norm,
        #         save_model_after_epoch,
        #         experiment_name=model_output_dir,
        #         output_dir=model_output_dir,
        #         model_type=model_type)

        # save_model(model, model_output_dir, model_output_dir, tokenizer=tokenizer)

        try:
            for isx in [train_path, valid_path]:
                os.remove(os.path.join(data_dir, isx.replace('csv','tsv')))
        except: pass





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

        config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[model_type]
        config = config_class.from_pretrained(model_name_or_path)


        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenizer = tokenizer.from_pretrained(model_output_dir + '/' + model_output_dir, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config = config)
        model = model.from_pretrained(model_output_dir + '/' + model_output_dir)


        model.to(device)


        torch.cuda.empty_cache()
        for frac in [0.1,0.2,0.3,0.4,0.5]:
            torch.cuda.empty_cache()
            frac = str(int(frac*100))
            if len(frac) == 1: frac = frac +'0'
            for rep in range(1,11):



                if os.path.exists('/home/mmoslem3/RES_folded/EMT_WDC_random_noise/EM_score_'+task.replace('folded','')+'_'+str(fold)+'_'+frac+'_'+str(rep)+'.csv'): continue
                df_original = pd.read_csv(data_dir+'test_00_'+str(rep)+'.csv')
                df = pd.read_csv(data_dir+'test_'+str(frac)+'_'+str(rep)+'.csv')
                


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


                # df.to_csv(data_dir+'test_noisyWord.csv',index=False)

                df['combined_left'] = df.filter(like='left').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
                df['combined_right'] = df.filter(like='right').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
                df = df[['id' , 'combined_left','combined_right','label']]
                df = df.rename(columns={'id':'idx','combined_left': 'text_left', 'combined_right': 'text_right','label':'label'})
                isx = "test_"+frac+'_'+str(rep)+".csv"  
                df.to_csv(os.path.join(data_dir, isx.replace('csv','tsv')),sep='\t', index=False)

        
                    
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
                df.to_csv('/home/mmoslem3/RES_folded/EMT_WDC_random_noise/EM_score_'+task.replace('folded','')+'_'+str(fold)+'_'+frac+'_'+str(rep)+'.csv', index=False)  

