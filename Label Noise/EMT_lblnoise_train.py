from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import torch
import random
import csv
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from src.data_loader import load_data, DataType
from src.data_representation import DeepMatcherProcessor
from src.evaluation import Evaluation
from src.model import save_model
from src.optimizer import build_optimizer
from src.training import train
from pytorch_transformers import BertTokenizer
from pytorch_transformers.modeling_bert import BertForSequenceClassification
from src.config import Config


NUM =2
gpu_no = str(NUM)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


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












for task in ['Fodors-Zagats','iTunes-Amazon','Walmart-Amazon']:
    for frac in [0.05,0.1,0.15,0.2,0.25]:

        dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
        embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
        data_dir = dataset_dir
        if 'Walmart' in task: R=[1,2]
        else: R =[1,2,3,4,5]

        for rep in R:


            df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/train.csv')
            num_to_flip = int(len(df) * frac)
            indices_to_flip = np.random.choice(df.index, size=num_to_flip, replace=False)
            df.loc[indices_to_flip, 'label'] = 1 - df.loc[indices_to_flip, 'label']
            df.to_csv(dataset_dir+'train2.csv',index=False)
            


            df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/valid.csv')
            num_to_flip = int(len(df) * frac)
            indices_to_flip = np.random.choice(df.index, size=num_to_flip, replace=False)
            df.loc[indices_to_flip, 'label'] = 1 - df.loc[indices_to_flip, 'label']
            df.to_csv(dataset_dir+'valid2.csv',index=False)






            num_epochs = 100
            model_type = 'bert' # bert
            model_name_or_path = 'bert-base-uncased' # "pre_trained_model/bert-base-uncased
            train_batch_size = 40
            eval_batch_size = 32
            max_seq_length = 256

            model_output_dir = "EMThierTrain" +task+str(frac*100+1)+"/"
            weight_decay = 0
            max_grad_norm = 1
            warmup_steps = 0 
            adam_eps = 1e-8
            learning_rate = 2e-5
            save_model_after_epoch = False
            do_lower_case = True



            test_path = 'test.csv'
            train_path = 'train2.csv'
            valid_path = 'valid2.csv'
            for isx in [test_path, train_path, valid_path]:
                df = pd.read_csv(data_dir + '/'+ isx )
                df['combined_left'] = df.filter(like='left').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
                df['combined_right'] = df.filter(like='right').apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
                df = df[['id' , 'combined_left','combined_right','label']]
                df = df.rename(columns={'id':'idx','combined_left': 'text_left', 'combined_right': 'text_right','label':'label'})
                df.to_csv(os.path.join(data_dir, isx.replace('csv','tsv')),sep='\t', index=False)




            device = 'cuda:'+str(NUM) if torch.cuda.is_available() else 'cpu'
            processor = DeepMatcherProcessor()
            label_list = processor.get_labels()
            config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[model_type]
            config = config_class.from_pretrained(model_name_or_path)
            tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
            model = model_class.from_pretrained(model_name_or_path, config=config)
            model.to(device)







            train_examples = processor.get_train_examples_fold2(data_dir,rep,frac) 
            training_data_loader = load_data(train_examples,label_list,tokenizer,max_seq_length,train_batch_size,DataType.TRAINING, model_type)


            num_train_steps = len(training_data_loader) * num_epochs
            optimizer, scheduler = build_optimizer(model,num_train_steps,learning_rate,adam_eps,warmup_steps,weight_decay)
            eval_examples = processor.get_dev_examples_fold2(data_dir,rep,frac) 
            evaluation_data_loader = load_data(eval_examples,label_list,tokenizer,max_seq_length,eval_batch_size,DataType.EVALUATION, model_type)


            evaluation = Evaluation(evaluation_data_loader, model_output_dir, model_output_dir, len(label_list), model_type)




            train(device,
                    training_data_loader,
                    model,
                    optimizer,
                    scheduler,
                    evaluation,
                    num_epochs,
                    max_grad_norm,
                    save_model_after_epoch,
                    experiment_name=model_output_dir,
                    output_dir=model_output_dir,
                    model_type=model_type)

            save_model(model, model_output_dir, model_output_dir, tokenizer=tokenizer)



    # ----------------------------------------v



            config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[model_type]
            config = config_class.from_pretrained(model_name_or_path)


            device = torch.device("cuda:"+str(NUM) if torch.cuda.is_available() else "cpu")

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            tokenizer = tokenizer.from_pretrained(model_output_dir + '/' + model_output_dir, do_lower_case=True)
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config = config)
            model = model.from_pretrained(model_output_dir + '/' + model_output_dir)
            model.to(device)
            torch.cuda.empty_cache()






            processor = DeepMatcherProcessor()
            test_batch_size = 32 
            test_examples = processor.get_test_examples(data_dir)
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
            if frac2 =='50': frac2='05'
        



            try: 
                os.mkdir('/home/mmoslem3/RES_hierarch/EMtransformer_'+task+'_'+'lblnoise')
            except:
                pass
            df.to_csv('/home/mmoslem3/RES_hierarch/EMtransformer_'+task+'_'+'lblnoise'+ '/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
            print(task + '_'+str(frac2)+'_'+str(rep))





