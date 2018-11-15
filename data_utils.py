import os
import re
import gensim
import numpy as np


LEX_PATH = os.path.expanduser('~/datasets/liu-bing/opinion-lexicon-English')
SENT_PATH = os.path.expanduser('~/datasets/sentence-sentiment-data/finegrained.txt')
DOC_PATH = os.path.expanduser('~/datasets/mdsd-v2')

word2vec_path = os.path.expanduser('~/datasets/word2vec/')


def is_better(word):
    if (
            '-' in word or word.endswith('ly') or
            word.endswith('er') or
            word.endswith('est') or
            word.endswith('s') or
            word.endswith('ing') or
            len(word) <= 2 or len(word) >= 9):
        return False
    else:
        return True


def load_lexicon(lexicon_path=LEX_PATH):
    print('loading lexicon: {}'.format(lexicon_path))
    positive_path = os.path.join(lexicon_path, 'positive-words.txt')
    negative_path = os.path.join(lexicon_path, 'negative-words.txt')
    with open(positive_path, 'r', encoding='utf8') as fp:
        positive_words = set(line.strip() for line in fp if not line.startswith(';') and is_better(line.strip()))
    with open(negative_path, 'r', encoding='utf8') as fn:
        negative_words = set(line.strip() for line in fn if not line.startswith(';') and is_better(line.strip()))
    conflicts = positive_words & negative_words
    for conflict in conflicts:
        positive_words.remove(conflict)
        negative_words.remove(conflict)
    print(' - positive words:', len(positive_words))
    print(' - negative words:', len(negative_words))
    print(' - total:', len(positive_words) + len(negative_words))
    hu_liu_lexicon = {word: 1 for word in positive_words}
    hu_liu_lexicon.update({word: 0 for word in negative_words})
    return hu_liu_lexicon


def load_sentences(domain):
    print('loading sentence dataset: {}'.format(domain))
    assert domain in {'books', 'dvds', 'electronics'}, 'domain must be one of books/dvds/electronics'
    pos_sents = []
    neg_sents = []
    in_domain = False
    doc_count = 0
    with open(SENT_PATH, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            if re.match(r'\w+_(pos|neg|neu)_\d+', line):
                current_domain = line.split('_')[0]
                if current_domain == domain:
                    in_domain = True
                    doc_count += 1
                else:
                    in_domain = False
            if in_domain:
                if line.startswith('pos'):
                    pos_sents.append(line[4:])
                if line.startswith('neg'):
                    neg_sents.append(line[4:])
    print(' - doc count:', doc_count)
    print(' - pos sents:', len(pos_sents))
    print(' - neg sents:', len(neg_sents))
    sents = pos_sents + neg_sents
    labels = [1] * len(pos_sents) + [0] * len(neg_sents)
    return sents, labels


def load_documents(domains, n_unlabeled=None, n_labeled=None, verbose=False):
    if n_unlabeled is None:
        n_unlabeled = 0
    sorted_data_path = os.path.join(DOC_PATH, 'sorted_data')
    print('loading data from {}'.format(sorted_data_path))
    texts = []
    s_labels = []
    d_labels = []
    sentiments = ('positive', 'negative', 'unlabeled') if n_unlabeled else ('positive', 'negative')
    for d_id, d_name in enumerate(domains):
        for s_id, s_name in zip((1, 0, -1), sentiments):
            fpath = os.path.join(sorted_data_path, d_name, s_name + '.review')
            count = 0
            text = ''
            in_review_text = False
            with open(fpath, encoding='utf8', errors='ignore') as fr:
                for line in fr:
                    if '<review_text>' in line:
                        text = ''
                        in_review_text = True
                        continue
                    if '</review_text>' in line:
                        in_review_text = False
                        text = text.lower().replace('\n', ' ').strip()
                        text = re.sub(r'&[a-z]+;', '', text)
                        text = re.sub(r'\s+', ' ', text)
                        texts.append(text)
                        s_labels.append(s_id)
                        d_labels.append(d_id)
                        count += 1
                    if in_review_text:
                        text += line
                    # unlabeled cutoff
                    if (s_id == -1) and n_unlabeled and (count == n_unlabeled):
                        break
                    # labeled cutoff
                    if (s_id >= 0) and n_labeled and (count == n_labeled):
                        break
    print('data loaded')
    s_labels = np.asarray(s_labels, dtype='int')
    d_labels = np.asarray(d_labels, dtype='int')
    print(' - texts:', len(texts))
    print(' - s_labels:', len(s_labels))
    print(' - d_labels:', len(d_labels))

    return texts, s_labels, d_labels


def balanced_subsample2(x, y, subsample_num=2):
    """sub-sample `subsample_num` samples in which classes of `y` appeared approximately equally"""
    x = np.asarray(x)
    y = np.asarray(y)
    x_y0 = x[y == 0]
    x_y1 = x[y == 1]
    n0 = subsample_num // 2
    n1 = subsample_num - n0
    xs, ys = [], []
    for ci, this_xs, n in zip((0, 1), (x_y0, x_y1), (n0, n1)):
        if len(this_xs) > n:
            np.random.shuffle(this_xs)
        x_ = this_xs[:n]
        y_ = np.empty(n)
        y_.fill(ci)
        xs.append(x_)
        ys.append(y_)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return xs, np.asarray(ys, dtype=int)


def balanced_subsample3(x1, x2, y, subsample_num):
    """sub-sample `subsample_num` samples in which classes of `y` appeared approximately equally"""
    assert len(x1) == len(x2) == len(y)
    x1 = np.asarray(x1)
    y = np.asarray(y)
    x2 = np.asarray(x2)
    x_y0 = x1[y == 0]
    x_y1 = x1[y == 1]
    x1_y0 = x2[y == 0]
    x1_y1 = x2[y == 1]
    n0 = subsample_num // 2
    n1 = subsample_num - n0
    xs, x1s, ys = [], [], []
    for ci, this_xs, this_x1s, n in zip((0, 1), (x_y0, x_y1), (x1_y0, x1_y1), (n0, n1)):
        if len(this_xs) > n:
            # shuffle
            _indices = list(range(len(this_xs)))
            np.random.shuffle(_indices)
            this_xs = this_xs[_indices]
            this_x1s = this_x1s[_indices]
        x_ = this_xs[:n]
        x1_ = this_x1s[:n]
        y_ = np.empty(n)
        y_.fill(ci)
        xs.append(x_)
        x1s.append(x1_)
        ys.append(y_)
    xs = np.concatenate(xs)
    x1s = np.concatenate(x1s)
    ys = np.concatenate(ys)
    return xs, x1s, np.asarray(ys, dtype=int)


def load_word2vec_bin(path=word2vec_path, filename=None, vocab=None, verbose=False):
    """Loads 300x1 word vecs from Google (Mikolov) word2vec
    if `vocab` given (set or map), only words in `vocab` are loaded
    """
    if not filename:
        filename = 'GoogleNews-vectors-negative300.bin.gz'
    word2vec_bin_path = os.path.join(path, filename)
    print('loading word2vec vectors from', word2vec_bin_path)
    word2vec = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_bin_path, binary=True)
    for word in model.wv.vocab:
        if not vocab or (vocab and word in vocab):
            word2vec[word] = model[word]
    return word2vec


def get_embedding_mat(embeddings, word2index, embedding_dim, random_uniform_level=0.01, idx_from=2):
    """Use embeddings and word2index to get embedding-mat (for input layer)
    idx_from=2, usually, 0 for <PAD>, 1 for <OOV>
    """
    # embedding_mat = np.zeros((n_words, embedding_dim))
    n_words = len(word2index)
    for idx in range(0, idx_from):
        if idx in word2index.values():
            n_words -= 1
    n_words += idx_from
    embedding_mat = np.random.uniform(low=-random_uniform_level, high=random_uniform_level, size=(n_words, embedding_dim))
    embedding_mat[0] = np.zeros(embedding_dim)
    for word, idx in word2index.items():
        if idx < idx_from:
            continue
        embedding_vec = embeddings.get(word)
        if embedding_vec is not None:  # means we have this word's embedding
            embedding_mat[idx] = embedding_vec
    return embedding_mat


if __name__ == '__main__':
    pass
