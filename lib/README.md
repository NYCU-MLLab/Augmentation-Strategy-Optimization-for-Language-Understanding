![downloads](https://img.shields.io/github/downloads/atom/atom/total.svg)
![build](https://img.shields.io/appveyor/ci/:user/:repo.svg)
![chat](https://img.shields.io/discord/:serverId.svg)

## \_\_init\_\_

\_\_init\_\_.py is responsible for building a python module for importing usage.

In default, the file is ***empty***.

## augmenter

The augmenter file contains the class about augmenting the document. All the augmenter is intantiated from base class, ***Augmenter***.

### Augmenter

Base class for the other augmentation methods. It contains a pure virtual function waiting for overriding by child class.

#### _action
![](https://i.imgur.com/El3U37H.png)

### DeleteAugmenter

Responsible for deleting augmentation method.

### SwapAugmenter

Responsible for swapping augmentation method.

### ReplaceAugmenter

Responsible for replacing augmentation method.

### InsertAugmenter

Responsible for inserting augmentation method.

### BackTransAugmenter

Responsible for backtranslation augmentation method.
![](https://i.imgur.com/c19Yyom.png)

The model name can infer from ***Huggingface***

### IdentityAugmenter

Responsible for stopping the augmentation process.

## classifier

Pytorch module for classification usage only. It can be specialized for complicated usage.

![](https://i.imgur.com/DfB7n6D.png)

For Hugginface usage, just a simple dense network.

## configurer

Configurer file is responsible for setting the hyperparameter which defined in ***model_config.json***. It will set the parameter and obtain the instanclize class.

![](https://i.imgur.com/A0fVjT9.png)

## dataset

Dataset file is responsible for preprocessing the ***csv*** datafile into the defined formatting provided by ***allennlp***.

### SentimentDatasetReader

SentimentDatasetReader can deal with all the csv file with format such as:

![](https://i.imgur.com/v9rJRQR.png)

### StanfordSentimentTreeBankDatasetReader

StanfordSentimentTreeBankDatasetReader is especaiily designed for SST dataset. It can create SST2 and SST5 dataset.

## embedder

The embedder file is responsible for embedding the documenter to embedding. 

### TextEmbedder

Interface for ***allennlp module***.
![](https://i.imgur.com/KzvDXWc.png)

### UniversalSentenceEmbedder

Noticed that this embedder is a ***tensorflow module***.
![](https://i.imgur.com/rkXe6nD.png)

## encoder

The encoder is responsible for encoding the embedding once again.

It is defined for general usage. In standard NLP pipeline, the word will be first embeded from probable ***Glove, and W2V***. And it will be further encoded by additional feedfoward network.

![](https://i.imgur.com/EXoIKoI.png)

For Transformer-based embedding, it will not do anything.

![](https://i.imgur.com/HDNe9Vu.png)


## loss

Loss file is responsible for calculating the loss for model training. It supports JsdCrossEntropy, EntropyLoss, IBLoss, and SupConLoss.

All loss can be directly use in the pytorch process.

![](https://i.imgur.com/yyRjNE5.png)

## model

Model file will provide training of different pre-defined module. All the training related module will be intergrated here.

![](https://i.imgur.com/tBPqtmb.png)
![](https://i.imgur.com/r4zDnZL.png)


## reinforcer

Reinforcer file is responsible for the training of our augmentation policy. It contains two small module. Enviorment and Policy.

### Reinforcer
![](https://i.imgur.com/so6rdns.png)

### Enviroment
The enviroment will input augmented document and output reward for the overall training.

![](https://i.imgur.com/PyOFirv.png)

### Policy
The policy will output an action.
![](https://i.imgur.com/y3HJHrm.png)

## tokenizer

Tokenizer file will incharge for detokenizing and tokenizing. Noticed that the augmentation methods is a word-based augmentation. So sub-word need to detokenize back to original word.

![](https://i.imgur.com/eZvpA4l.png)

## trainer

Trainer file will responsible for training. Including the multipler of different loss term, gradient accumulation ...

All the trainer will intantiate from 

![](https://i.imgur.com/ICGCfph.png)

### ReinforceTrainer

![](https://i.imgur.com/B3JDo0X.png)

### TextTrainer

![](https://i.imgur.com/mWrOXQh.png)

## utilis

Utilis file is responsible for small operation such as load object, save object, and zip or unzip some data structure.

![](https://i.imgur.com/x2uXQIY.png)
