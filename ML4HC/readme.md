## AI612: Machine Learning for Healthcare  

### Assignment1: ICU Mortality Prediction  
- Predicted mortality following the guideline of Assignment1  
- Used GRU & logistic regression to measure performance   
- Didn't upload 6 data files due to file size issue in github.  

  ```  
  * Droped Files * 
  X_train_rnn.npy
  X_train_logistic.npy
  y_train.npy
  X_test_rnn.npy
  X_test_logistic.npy
  y_test.npy
  ```  
---  

### Assignment2: Automatic Coding  
- Predict about 19K ICD9 Dignosis/Procedure codes (Multi-label classification)  
- Free to choose the model (Reference Baseline: CAML ([Explainable Prediction of Medical Codes from Clinical Text](https://arxiv.org/abs/1802.05695)))  
- Mine: Bio Clinical Bert + 1D CNN + Attention (But, got poor performance)  
- Didn't upload 7 data files due to file size issue in github.  

  ```  
  * Droped files *
  - X_train.pickle  
  - X_test.pickle
  - y_train.txt
  - y_test.txt
  - trained_model_HC2.pt # weights for my model 
  - test_index.pickle # Data re-processing for 'macro' auroc estimation  
  - label_list.txt # Whole ICD9 codes in MIMIC3
  ```  
  
  - Process  

    ```  
    # Data processing => Get X_train.pickle, X_test.pickle, y_train.txt, y_test.txt, test_index.pickle, label_list.txt  
    python ./data/preprocess.py  
    
    # Get weight => Model is trained and get trained_model_HC2.pt  
    python main.py --assignment_mode train  
    
    # Test  
    python main.py
    ```  
    
