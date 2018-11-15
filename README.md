# HSN for sentence-level sentiment classification

HSN model proposed in paper *"Neural Sentence-level Sentiment Classification with Heterogeneous Supervision"*, which is accepted by [ICDM 2018](http://icdm2018.org) as a short paper.

## Data

1. Sentence-level dataset: [Finegrained Sentiment Dataset](https://github.com/oscartackstrom/sentence-sentiment-data)
2. Document-level dataset: [Multi-Domain Sentiment Dataset (version 2.0)](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/). The dataset we used is [unprocessed.tar.gz](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz) which contains the original data.

3. Sentiment Lexicon (i.e. word-level): [Bing Liu's Opinion Lexicon (or Sentiment Lexicon)](http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)

Detailed usages of these datasets are introduced in the paper.

## Train & Test

For example, you can use the following command to train and test HSN model:

> python train.py --domain dvds

**Notice:** You should first download these datasets and `word2vec` pre-trained using Google News corpus [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) to run this code.

## Code Description

 - `data_utils.py` data and embedding utils
 - `model.py` the HSN model, train and test included
 - `main.py` main file to run this code


For any issues, you can contact me via [yuanzhigang10@163.com](mailto:yuanzhigang10@163.com)
