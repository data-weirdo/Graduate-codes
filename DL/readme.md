## AI502: Deep Learning  

### Assignment1: Image Classification using Celeba dataset  
- Experiment with 36 combinations  
- Didn't upload `Celeba` image files due to file size issue in github.  
  
  ![Table1](https://user-images.githubusercontent.com/43376853/119230264-6d48c800-bb56-11eb-86be-e015677f5b73.png)

### Assignment2: QA using squad  
- Ordered to use 3-models   

  ```  
  1. LSTM
  2. Transformer
  3. BERT-mini 
  ```  
  
  ![Result](https://user-images.githubusercontent.com/43376853/122561637-0ea33b00-d07d-11eb-8486-4e39b2bfb4b5.png)

  - 1. LSTM에서 감점 당했는데, F1 score가 잘 나오지 않았던 이유 
    -> LSTM에 넣을 때 길이로 sorting을 하고 lstm에 넣었는데, 다시 원래대로 순서를 복원하지 않았음...!!!  
