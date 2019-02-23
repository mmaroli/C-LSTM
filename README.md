# C-LSTM Python Implementation

A basic python implementation of [A C-LSTM Neural Network for Text Classification](https://arxiv.org/pdf/1511.08630.pdf)

Description of [Yelp Dataset](https://www.yelp.com/dataset/documentation/main)

Files Present:
  + create_embedding.py
  + create_train_dev_split.py
  + clstm.py

Run clstm.py to train model.
  1) docker-compose up --build -d
  2) docker exec -it "clstm-model-container" python clstm.py
