# Pytorch-Neural-Tensor-Network
This is pytorch implementation of [Neural Tensor Network (NTN)](http://ijcai.org/Proceedings/15/Papers/329.pdf) with Attention Model

### Dataset
To test model, I use a dataset of 50,000 movie reviews taken from IMDb. 
It is divied into 'train', 'test' dataset and each data has 25,000 movie reviews and labels(positive, negetive).
You can access to dataset with this [link](http://ai.stanford.edu/~amaas/data/sentiment/)

### Reference
This project is highly depend on "SVO extractor". Thanks to wonderful [work](https://github.com/peter3125/enhanced-subject-verb-object-extraction) from "peter3125" which can extract triple set(subject, object, verb) very easily, I can make pipeline by including his open source. So someone who try this project should contact his repository to download resource.

### How to use it?
Follow the example

#### 1 Download "SVO extractor"
Check this [link](https://github.com/peter3125/enhanced-subject-verb-object-extraction) to download "SVO extractor" and make folder "extractor" to put downloaded resources.

#### 2 Generate Word2Vec Embeddings
I implement "gensim" library to generate Word2vec embeddings. To generate word2vec embeddings, follow the sample

```python
python word_embeder.py --train_path source/train.csv --dict_path word2vec --tokenizer_name word_tokenizer --size 200 --window 5 --min_count 3
```

#### 3 Train Model
There is a lot of options to check.
1. train_path : A File to train model
2. valid_path : A File to valid model
3. dict_path : A Path of Word2vec model for embeddings of HAN model
4. save_path : A Path to save result of HAN model
5. max_sent_len : Maximum length of sentence to analysis ( Sentences of each document which is exceed to the limit is eliminated to train model )
6. max_svo_len : Maximum length of word for each SVO(subject/object/verb) to analysis ( Words of each sentence which is exceed to the limit is eliminated to train model )
7. tensor_dim : A Tensor size of model
8. atten_size : A Attention size of model
9. hidden_size : A hidden size of GRU model
10. n_layers : A number of layers of GRU model
11. n_epochs : A number of epoches to train
12. dropout_p : dropout probability
13. lr : learning rate
14. early_stop : A early_stop condition. If you don't want to use this options, put -1
15. batch_size : Batch size to train

```python
python train.py --train_path source/train.csv --valid_path source/test.csv --dict_path word2vec/1 --hidden_size 256 --atten_size 128 --batch_size 16
```

### Result
Result with hyper parameter settings

| word2vec dimention | hidden size | atten size | tesor size | Best Epoch |  lr  | train loss | valid loss | valid accuracy |
|--------------------|:-----------:|:----------:|:----------:|:----------:|:----:|:----------:|:----------:|:---------------|
| 200                |     64      |     64     |     64     |      1     |0.0001|   0.0396   |   0.0392   |     0.6459     |
| 200                |    128      |    128     |    128     |      1     |0.0001|   0.0394   |   0.0385   |     0.6584     |


### Repo available online
1. Neural Tensor network
    - [Reimplementing Neural Tensor Networks for Knowledge Base Completion (KBC) in the TensorFlow framework](https://github.com/dddoss/tensorflow-socher-ntn)
2. Stock Prediction with Deep Learning
    - [Event-Driven-Stock-Prediction-using-Deep-Learning](https://github.com/vedic-partap/Event-Driven-Stock-Prediction-using-Deep-Learning)
    - [Sentiment-Analysis-in-Event-Driven-Stock-Price-Movement-Prediction](https://github.com/WayneDW/Sentiment-Analysis-in-Event-Driven-Stock-Price-Movement-Prediction)
3. Text Classification with CNN
    - [cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch)
    
### Reference

    
### Comment
We got bad result with Neural Tensor Network(NTN) compare to other works like [HAN](https://github.com/JoungheeKim/Pytorch-Hierarchical-Attention-Network), [BERT](https://github.com/JoungheeKim/Pytorch-BERT-Classification). The reason we can find easily is that this model is highly depend on the 'SVO extractor'. If 'SVO extractor' is not good enough to catch valuable SVO(Subject/Verb/Object) in the sentences, other model in the pipeline will not guarante good result neither. Another reason we can guess is insufficient data to train  model. NTN model have a lot of parameters, because of unique structure(bidirectional batch multiplication). And NTN model decrease data from dataset to extract inputs, because of unique input structure(Subject/Verb/Object). Above reasons make it hard to get good result.


You can see the [detail review](https://github.com/JoungheeKim/Pytorch-Neural-Tensor-Network/blob/master/REVIEW.md) of mine, if you are korean.



