import pickle
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from collections import OrderedDict
from transformers import PreTrainedTokenizer


class DischargeSummaryDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, identifier: str):
        """
        * identifier
        'train' : read X_train.pickle, ./y_train.txt
        'test' : read X_test.pickle, ./y_test.txt
        """
        self.tokenizer = tokenizer
        self.file = []
        self.feature_list = []

        path = [f'./X_{identifier}.pickle', f'./y_{identifier}.txt']

        for file_index, file_path in enumerate(path):
            if file_index == 0:
                with open(file_path, 'rb') as f:
                    temp = pickle.load(f)
            else:
                with open(file_path, 'r') as f:
                    temp = f.readlines()
            self.file.append(temp)
            f.close()

        self.X, self.y = self.file
        self.y = self.data_key_mapping(self.X, self.y)
        self.clinical_bert_data(self.X, self.y)

    def data_key_mapping(self, feature_dat, label_dat):
        ordered_label_dict = OrderedDict()
        od_keys = list(feature_dat.keys())
        label_dat = self.label_reprocess(label_dat)

        for x_idx in od_keys:
            for y_key, values in label_dat.items():
                if  x_idx == y_key:
                    ordered_label_dict[y_key] = values
                    break

        return ordered_label_dict

    def label_reprocess(self, label_data):
        """
        Assignment's pre-defined label format: '.txt' with ['100001,0,0,0,0,0,0,...,0\n','100003,0,0,0,0,0,0,...,0\n', ..., '...,0'] 
        I used f.writelines to save the results => Inevitabliy '\n' was added at the end of each line. 
        """
        y_dict = dict()

        for contents in label_data:
            key = contents.split(',')[0]
            values = contents.split(',')[1:]
            # It might have been better to use 'readline' than manually changing it by using 'readlines'. 
            if values[-1] == '0\n':
                values[-1] = '0'
            if values[-1] == '1\n':
                values[-1] = '1'
            values = list(map(lambda x: int(x), values))
            y_dict[key] = values

        return y_dict

    def clinical_bert_data(self, feature_dat, label_dat):
        for key, values in feature_dat.items():
            tokenized_values = self.tokenizer(values)
            tokenized_values['labels'] = label_dat[key]
            self.feature_list.append(tokenized_values)

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, index):
        return self.feature_list[index]