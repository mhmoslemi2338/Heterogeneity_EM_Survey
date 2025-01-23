import numpy as np
import pandas as pd
import torch
import random
import csv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
from src.data_loader import load_data, DataType
from src.data_representation import DeepMatcherProcessor
from tqdm import tqdm
from torch.utils import data
from pytorch_transformers import BertTokenizer
from pytorch_transformers.modeling_bert import BertForSequenceClassification
from src.data_representation import DeepMatcherProcessor
from src.data_loader import load_data, DataType
from src.config import Config


gpu_no = "2"
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






def clac_p_berno(val):
    MISS_COND = {1 : 0.8, 0: 0.2}
    x = MISS_COND[val['label']]
    return (np.arctan(x)/np.arctan(len(MISS_COND)))+ random.uniform(-0.05, 0.05)



def inject_MCAR(df_in,frac,cols):
    df = df_in.copy()

    n_missing = int(frac * len(df) * len(cols))
    missing_indices = []
    while len(missing_indices) < n_missing:
        row = np.random.randint(0, df.shape[0])
        col = np.random.randint(0, len(cols))
        missing_indices.append(np.array([row, col]))
        missing_indices  = list(np.unique(missing_indices,axis = 0))
    for row, col in missing_indices:
        df.loc[row,'left_'+cols[col]] = np.nan

    n_missing = int(frac * len(df) * len(cols))
    missing_indices = []
    while len(missing_indices) < n_missing:
        row = np.random.randint(0, df.shape[0])
        col = np.random.randint(0, len(cols))
        missing_indices.append(np.array([row, col]))
        missing_indices  = list(np.unique(missing_indices,axis = 0))
    for row, col in missing_indices:
        df.loc[row,'right_'+cols[col]] = np.nan
    return df



def inject_MAR(df_in,frac,cols):
    df = df_in.copy()
    left_cols = ['left_'+col for col in cols]
    right_cols = ['right_'+col for col in cols]

    n_missing = int(frac * len(df) * len(cols))
    number_nans = df.isna()[left_cols].sum().sum()
    while number_nans < n_missing:
        col = np.random.randint(0, len(cols))
        row = np.random.randint(0, df.shape[0])
        p_berno  = clac_p_berno(df.loc[row])
        action = [np.random.rand() < p_berno for _ in range(1)][0]
        if action : 
            df.loc[row,'left_'+cols[col]] = np.nan
        number_nans = df.isna()[left_cols].sum().sum()

    n_missing = int(frac * len(df) * len(cols))
    number_nans = df.isna()[right_cols].sum().sum()
    while number_nans < n_missing:
        col = np.random.randint(0, len(cols))
        row = np.random.randint(0, df.shape[0])
        p_berno  = clac_p_berno(df.loc[row])
        action = [np.random.rand() < p_berno for _ in range(1)][0]
        if action : 
            df.loc[row,'right_'+cols[col]] = np.nan
        number_nans = df.isna()[right_cols].sum().sum()

    return df








def clac_p_berno_MNAR(val,COL):
    MISS_COND = {1 : 0.8, 0: 0.2}
    x = MISS_COND[val['label']] 
    x+= (hash(val[COL])% 100)/100
    return (np.arctan(x)/np.arctan(len(MISS_COND)+100)) + random.uniform(-0.05, 0.05)


def inject_MNAR(df_in,frac,cols):
    df = df_in.copy()
    left_cols = ['left_'+col for col in cols]
    right_cols = ['right_'+col for col in cols]

    n_missing = int(frac * len(df) * len(cols))
    number_nans = df.isna()[left_cols].sum().sum()
    while number_nans < n_missing:
        col = np.random.randint(0, len(cols))
        row = np.random.randint(0, df.shape[0])
        p_berno  = clac_p_berno_MNAR(df.loc[row],'left_'+cols[col])
        action = [np.random.rand() < p_berno for _ in range(1)][0]
        if action : 
            df.loc[row,'left_'+cols[col]] = np.nan
        number_nans = df.isna()[left_cols].sum().sum()
    
    n_missing = int(frac * len(df) * len(cols))
    number_nans = df.isna()[right_cols].sum().sum()
    while number_nans < n_missing:
        col = np.random.randint(0, len(cols))
        row = np.random.randint(0, df.shape[0])
        p_berno  = clac_p_berno_MNAR(df.loc[row],'right_'+cols[col])
        action = [np.random.rand() < p_berno for _ in range(1)][0]
        if action : 
            df.loc[row,'right_'+cols[col]] = np.nan
        number_nans = df.isna()[right_cols].sum().sum()
    
    return df







for task in ['Fodors-Zagats','iTunes-Amazon','Walmart-Amazon']:
    for MODE in ['MCAR','MAR','MNAR']:



        for frac in [0.1,0.2,0.3,0.4,0.5]:
            for rep in range(1,11):

                df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/test.csv')
                cols = []
                for col in df.columns:
                    col = col.replace('left_','').replace('right_','')
                    if col not in cols:
                        if col not in ['id','label']:
                            cols.append(col)


                if MODE == 'MCAR':
                    df = inject_MCAR(df,frac,cols)
                elif  MODE =='MAR':
                    df = inject_MAR(df,frac,cols)
                else:
                    df = inject_MNAR(df,frac,cols)

                            


                model_path = 'VLDB_best_model_'+task+'.pth'
                dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
                embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
                df.to_csv(dataset_dir+'test_noisyWord_miss.csv',index=False)



                data_dir = dataset_dir
                for isx in ['test_noisyWord_miss.csv']:
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
                model_output_dir = 'vldb_MODEL_' + task
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
                test_batch_size = 64
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


                try: 
                    os.mkdir('/home/mmoslem3/RES_hierarch/EMtransformer_'+task+'_'+MODE)
                except:
                    pass
                df.to_csv('/home/mmoslem3/RES_hierarch/EMtransformer_'+task+'_'+MODE+ '/DM_score_'+task+'_'+frac2+'_'+str(rep)+'.csv', index=False)  
                print(MODE + '_'+str(frac2)+'_'+str(rep))



