import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, BatchEncoding
from typing import List, Dict, Any, NewType


@dataclass
class HFDataCollator:
    """
    Data collator need for clinical bert pre-processing
    """
    tokenizer: PreTrainedTokenizer

    def _pad(self, encoded_inputs: Dict, max_length):
        current_input = encoded_inputs['input_ids']
        pad_needed = len(current_input) < max_length
        cut_needed = len(current_input) >= max_length

        if pad_needed:
            pad_length = max_length - len(current_input)
            encoded_inputs['input_ids'] = current_input + [self.tokenizer.pad_token_type_id] * pad_length
            encoded_inputs['attention_mask'] = [1]*len(current_input) + [0]*pad_length
            # encoded_inputs['token_type_ids'] += [self.tokenizer.pad_token_type_id] * pad_length 

        elif cut_needed:
            encoded_inputs['input_ids'] = current_input[:max_length]
            encoded_inputs['attention_mask'] = [1]*max_length
            # encoded_inputs['token_type_ids'] = encoded_inputs['token_type_ids'][:max_length]

        return encoded_inputs

    def __call__(self, feature_label_dat: List):
        feature_dat = []
        label_dat = []

        for fl_dat in feature_label_dat:
            label_instance = fl_dat['labels']
            label_dat.append(label_instance)
            feature_dat.append({k: v for k, v in fl_dat.items() if (k != 'labels') and (k in ['input_ids', 'attention_mask'])})

        # max_length = max([len(feature.input_ids) for feature in feature_dat])
        # BERT's max sequence length
        max_length = 512

        feature_batch = {}
        for encoded_inputs in feature_dat:
            outputs_padded = self._pad(encoded_inputs, max_length) 
            for key, values in outputs_padded.items():
                if key not in feature_batch:
                    feature_batch[key] = []
                feature_batch[key].append(values)
        feature_batch = {
            key: torch.tensor(feature_batch[key]) for key in feature_batch.keys()
        }

        feature_batch = BatchEncoding(feature_batch)
        label_dat = torch.FloatTensor(label_dat)
        
        return feature_batch, label_dat