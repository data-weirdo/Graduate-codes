import time
import re
import copy
import sys
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict


def main():

    start = time.time()

    noteevents= pd.read_csv('../MIMIC3_Raw/NOTEEVENTS.csv.gz', compression='gzip') # Main table
    icd_diag = pd.read_csv('../MIMIC3_Raw/D_ICD_DIAGNOSES.csv.gz', compression='gzip') # The table which contains whole icd9 codes related to 'DIAGNOSIS'
    icd_proc = pd.read_csv('../MIMIC3_Raw/D_ICD_PROCEDURES.csv.gz', compression='gzip', dtype={'ICD9_CODE': object}) # The table which contains whole icd9 codes related to 'PROCEDURES'
    diagnoses_icd = pd.read_csv('../MIMIC3_Raw/DIAGNOSES_ICD.csv.gz', compression='gzip') # The table which maps HADM_ID with DIAGNOSIS ICD9 Codes
    procedures_icd = pd.read_csv('../MIMIC3_Raw/PROCEDURES_ICD.csv.gz', compression='gzip', dtype={'ICD9_CODE': object}) # The table which maps HADM_ID with Procedure ICD9 Codes
    
    ##################################
    #### NOTEEVENTS table related ####
    ##################################

    noteevents = noteevents.loc[noteevents['HADM_ID'].isnull() != True, :]
    noteevents = noteevents.loc[noteevents['CATEGORY']=='Discharge summary', :]
    noteevents = noteevents.loc[noteevents['DESCRIPTION'] == 'Report', :]
    noteevents = noteevents.drop_duplicates(subset=['HADM_ID'], keep='last') # Deleted the case of 'HADM_ID' occuring more than two times
    noteevents['HADM_ID'] = noteevents['HADM_ID'].apply(lambda x: int(x))

    # Train / test split
    train = noteevents.loc[list(map(lambda x: str(x)[-1] not in ['8','9'], noteevents['HADM_ID'])),:]
    test = noteevents.loc[list(map(lambda x: str(x)[-1] in ['8','9'], noteevents['HADM_ID'])),:]

    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    # Whole HADM_IDs list which appear in train & test set
    in_train_test_hadm_id = list(train['HADM_ID']) + list(test['HADM_ID']) 

    #######################################
    #### D_ICD_DIAGNOSES table related ####
    #######################################

    whole_diag_icd9_list = list(icd_diag['ICD9_CODE'].values)
    whole_diag_icd9_list = list(map(lambda x: 'd'+x,  whole_diag_icd9_list)) # added 'd' to distingush the ICD9_codes from Procedure ICD9 codes

    ########################################
    #### D_ICD_PROCEDURES table related ####
    ########################################

    whole_proc_icd9_list = list(icd_proc['ICD9_CODE'].values)
    whole_proc_icd9_list = list(map(lambda x: 'p'+x,  whole_proc_icd9_list)) # added 'P' to distingush the ICD9_codes from Procedure ICD9 codes


    whole_label_candidate = whole_diag_icd9_list + whole_proc_icd9_list # Whole ICD9 codes in MIMIC3

    ############################################
    #### DIAGNOSES_ICD.csv.gz table related ####
    ############################################

    diagnoses_icd['ICD9_CODE'] = diagnoses_icd['ICD9_CODE'].apply(lambda x: 'd'+str(x)) # due to mixed types 

    # Key(HADM_ID), Value(ICD9 Code list per HADM_ID)
    diagnoses_hadm_icd9 = dict()
    for key, group in diagnoses_icd.groupby(['HADM_ID']):
        diagnoses_hadm_icd9[key] = group['ICD9_CODE'].values.tolist()


    #############################################
    #### PROCEDURES_ICD.csv.gz table related ####
    #############################################

    procedures_icd['ICD9_CODE'] = procedures_icd['ICD9_CODE'].apply(lambda x: 'p'+x)
    
    # Key(HADM_ID), Value(ICD9 Code list per HADM_ID)
    procedures_hadm_icd9 = dict()
    for key, group in procedures_icd.groupby(['HADM_ID']):
        procedures_hadm_icd9[key] = group['ICD9_CODE'].values.tolist()


    ###########################################################################################################
    ###########################################################################################################
    

    # Combine two dictionaries in same key 
    hadm_icd9 = copy.deepcopy(diagnoses_hadm_icd9)
    base_key = list(diagnoses_hadm_icd9.keys())

    for key, values in procedures_hadm_icd9.items():
        if key in base_key:
            hadm_icd9[key].extend(values)
        else:
            hadm_icd9[key] = values

    # Extract the hadm_ids which actually appears in train / test dataset
    hadm_icd9_real = dict()
    for key, values in hadm_icd9.items():
        if key in in_train_test_hadm_id:
            hadm_icd9_real[key] = values

    hadm_icd9 = hadm_icd9_real

    # train / test split
    train_hadm_dict = {}
    test_hadm_dict = {}
    err_cnt = 0

    for key, values in hadm_icd9.items():
        if key in list(train['HADM_ID']):
            train_hadm_dict[key] = values
            
        elif key in list(test['HADM_ID']):
            test_hadm_dict[key] = values
            
        else:
            err_cnt += 1

    assert err_cnt == 0, 'Error happened. Preprocessing is stopped.'

    # record multi-label data
    train_multi_label = multi_label(train_hadm_dict, whole_label_candidate)
    test_multi_label = multi_label(test_hadm_dict, whole_label_candidate)

    ##########
    ## Save ##
    ##########

    # discharge_summary_preprocess(train, 'X_train.pickle')
    discharge_summary_preprocess(test, 'X_test.pickle')

    # with open('./label_list.txt', 'w') as f:
    #     f.writelines(' '.join(whole_label_candidate))
    # f.close()

    to_txt(train_multi_label, 'y_train_temp.txt')
    to_txt(test_multi_label, 'y_test_temp.txt')

    # To calculate 'macro' auroc / auprc -> I need to pick the indice which has once appeared on test set
    testtime_macro_calculation_index_selection('./test_index.pickle')


    end = time.time()
    total_time = (end-start)/60
    print(f'Total time spent: {total_time} minute')

    return total_time

    # X_train, X_test format: OrderedDict
    # For example, [OrderedDict([('167118', "Discharge Diagnosis: C0PD ... as needed."), ('196489', "Discharge Diagnosis: ... (Daily)."), ...))])]

    # y_train, y_test format: List 
    # For example, ['100001,0,0,0,0,0,0,...0\n','100003,0,0,0,0,0,0,...0\n', ..., '...,0']


# In Discharge Summary code, there are too many '\n' and '\n\n' special characters are entangled.
# I inevitably should use regular expression, but '.' in regular expression quits when it meats '\n'.
# So, to Iron out the process, I made some_tweak function which removes the unnecessary extra effort for preprocessing. 
def some_tweak(data):
    df = data.replace('\n\n', '★')
    df = df.replace('\n', ' ')
    df = df.replace('★', '\n')  
    
    return df

def discharge_summary_preprocess(data, filename):

    data_ht = data[['HADM_ID', 'TEXT']]
    data_ht['HADM_ID'] = data_ht['HADM_ID'].apply(lambda x: str(x))
    data_ht['TEXT'] = data_ht['TEXT'].apply(lambda x: some_tweak(x))
    data_ht_len = len(data_ht)

    service_reg = '(Service:)(.*)'
    allergies_reg = '(Allergies:)(.*)'
    allergies_reg_upper = '(ALLERGIES:)(.*)'
    pmh_reg = '(Past Medical History:)(.*)'
    pmh_reg_upper = '(PAST MEDICAL HISTORY:)(.*)'
    discharge_diag_reg = '(Discharge Diagnosis)(.*)'
    discharge_diag_reg_upper = '(DISCHARGE DIAGNOSIS)(.*)'
    discharge_medi_reg = '(Discharge Medications)(.*)'
    discharge_medi_reg_upper = '(DISCHARGE MEDICATIONS)(.*)'

    text_dict = OrderedDict()

    for index in range(data_ht_len):
        table = data_ht.loc[index,['HADM_ID', 'TEXT']]
        hadm_id = str(table.HADM_ID) # would be a 'key'
        text = table.TEXT

        empty_string = ''

        if (re.search(discharge_diag_reg, text) != None) or (re.search(discharge_diag_reg_upper, text) != None):
            if re.search(discharge_diag_reg, text) != None:
                dis_diag = 'discharge diagnosis:' + re.search(discharge_diag_reg, text).group(2)
            else:
                dis_diag = 'discharge diagnosis:' + re.search(discharge_diag_reg_upper, text).group(2)
            empty_string = empty_string + dis_diag + ' [SEP] '

        if (re.search(pmh_reg, text) != None) or (re.search(pmh_reg_upper, text) != None):
            if re.search(pmh_reg, text) != None:
                pmh = 'past medical history:' + re.search(pmh_reg, text).group(2)
            else:
                pmh = 'past medical history:' + re.search(pmh_reg_upper, text).group(2)
            empty_string = empty_string + pmh + ' [SEP] '

        if re.search(service_reg, text) != None:
            service = re.search(service_reg, text).group(0)
            service = service.lower()
            empty_string = empty_string + service + ' [SEP] '

        # 1. There's no 'Allergies' or 'ALLERGIES' or 'allergies' in Bio Clinical BERT vocab.
        # 2. I should cut information that exceeds max length of BERT model (512).
        # So I decided not to include Allergy related text in the model. 

        # if (re.search(allergies_reg, text) != None) or (re.search(allergies_reg_upper, text) != None):
        #     if re.search(allergies_reg, text) != None:
        #         allergies = re.search(allergies_reg, text).group(0)
        #     else:
        #         allergies = 'Allergies:' + re.search(allergies_reg_upper, text).group(2)
        #     empty_string = empty_string + allergies + ' [SEP] '

        if (re.search(discharge_medi_reg, text) != None) or (re.search(discharge_medi_reg_upper, text) != None):
            if re.search(discharge_medi_reg, text) != None:
                dis_medi = 'discharge medications:' + re.search(discharge_medi_reg, text).group(2)
            else:
                dis_medi = 'discharge medications:' + re.search(discharge_medi_reg_upper, text).group(2)
            empty_string = empty_string + dis_medi

        text_dict[hadm_id] = empty_string

    with open(f'./{filename}', 'wb') as f:
        pickle.dump(text_dict, f)
    f.close()

    return None

def multi_label(data, whole_label):
    result = dict()
    
    for key, value in data.items():
        compare = np.array(value)
        criteria = np.array(whole_label)
        
        isin = np.isin(criteria, compare)
        temp = np.where(isin, 1, 0).tolist()
        
        result[key] = temp
        
    return result

def to_txt(data, filename):
    suggested_format = []

    for i, (key, values) in enumerate(data.items()):
        values = list(map(lambda x:str(x), values))
        values_to_list = ','.join(values)
        values_to_list = str(key) + ',' + values_to_list + '\n'
        suggested_format.append(values_to_list)

    with open(f'./{filename}', 'w') as f:
        f.writelines(suggested_format)
    f.close()

def testtime_macro_calculation_index_selection(outdir):
    with open('./y_test.txt') as f:
        data = f.readlines()
    f.close()

    temp = [0] * 18449
    true_index = []

    for instance in data:
        values = instance.split(',')[1:]
        if values[-1] == ['0\n']:
            values[-1] = '0'
        elif values[-1] == ['1\n']:
            values[-1] = '1'

        for index, v in enumerate(values):
            if v == '1':
                temp[index] = 1

    for index, value in enumerate(temp):
        if value == 1:
            true_index.append(index)
    
    with open(outdir, 'wb') as f:
        pickle.dump(true_index, f)
    f.close()


if __name__ == '__main__':
    main()