import pickle
import pandas as pd
import os
from lgbm_feature_extract_from_str import feature_list_from_seq
import yaml
from tqdm import tqdm
import numpy as np

class LGBM_TE_model:
    def __init__(self, 
                 models_dir: str,
                 utr3: str = '',
                 utr5: str = ''):
        print(f'Loading models from {models_dir}')
        self.models = {}
        info_file = os.path.join(models_dir, 'info.txt')
        # open info file as yaml
        with open(info_file) as f:
            self.info = yaml.load(f, Loader=yaml.FullLoader)
        self.features_to_extract = self.info['features_to_extract']

        for file in os.listdir(os.path.join(models_dir, 'model')):
            model = pickle.load(open(os.path.join(models_dir, 'model', file), 'rb'))
            model_fold = int(file.split('_')[2].removesuffix('.pkl'))
            self.models[model_fold] = model
        
        self.utr3 = utr3
        self.utr5 = utr5

    def predict_TE(self, cds: str):
        full_sequence = self.utr5+cds+self.utr3
        extracted_features = feature_list_from_seq(self.features_to_extract, full_sequence, len(self.utr5), len(cds), len(self.utr3), len(full_sequence))
        value = 0.0
        for _, model in tqdm(self.models.items()):
            value+= model.predict(extracted_features)
        
        value /= len(self.models)
        return value

if __name__ == '__main__':

    model = LGBM_TE_model(
        '/work/09360/dayang/ls6/protein-generation/protein_diffusion/LGBM/LL_P5_P3_CF_AAF_3mer_freq_5', # From https://github.com/CenikLab/TE_prediction_baseline/tree/main/results/human/all_cell_lines/lgbm-LL_P5_P3_CF_AAF_3mer_freq_5/
        'AAAAATTTGGCGATACTTTAC',
        'AAAAATTTGGCGATACTTTAC')
    
    data = model.predict_TE('AAAAATTTGGCGATGGGTGTGTAACCCAAACTTTAC')
