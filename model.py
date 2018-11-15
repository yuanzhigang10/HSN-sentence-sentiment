"""
Sentence level sentiment classification

Architecture:
 * Common embeddings (Embedding) + Common sentiment classifier {FC -> FC}
  - sent: [Embedding -> BiRNN] -> {FC -> FC}
  - doc : [sent-encoder] -> {FC -> FC}
  - lex : Embedding -> FC -> {FC -> FC}
"""
from collections import Counter
from copy import deepcopy
import numpy as np
from nltk import word_tokenize, sent_tokenize
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from keras.models import Model
from keras import layers, callbacks
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences

from data_utils import load_sentences, load_lexicon, load_documents, \
    balanced_subsample2, balanced_subsample3, load_word2vec_bin, get_embedding_mat


CV = 2


agree_words = {'and', 'also', 'besides', 'furthermore', 'hence', 'because'}
disagree_words = {'but', 'however', 'while', 'though', 'although', 'nevertheless'}
agree_indices = set()
disagree_indices = set()


def my_tokenize(_sent):
    _words = word_tokenize(_sent.lower())
    for _idx, _word in enumerate(_words):
        if _word.isdigit():
            _words[_idx] = '<NUM>'  # replace number token with <NUM>
    return _words


def get_encoded_data(args):
    global agree_words, disagree_words, agree_indices, disagree_indices
    # %% load data
    # load sentence data
    sents, labels = load_sentences(domain=args.domain)

    # load sentiment lexicon
    lexicon = load_lexicon()
    pos_words = [word for word in lexicon if lexicon[word] == 1]
    neg_words = [word for word in lexicon if lexicon[word] == 0]
    lex_labels = [1] * len(pos_words) + [0] * len(neg_words)
    lex_word_seqs = pos_words + neg_words

    # load document data
    mdsd_domain = 'dvd' if args.domain == 'dvds' else args.domain
    doc_texts, doc_labels, _ = load_documents(domains=(mdsd_domain,))  # just one domain, ignore domain labels

    ## build vocabulary
    counter = Counter()
    word_seqs = []
    doc_word_seqs = []
    doc_word_sseqs = []
    # tokenize to words
    for sent in sents:
        word_seqs.append(my_tokenize(sent))  # [[w1, w2, ...], ...]
    for doc in doc_texts:
        doc_word_seqs.append(my_tokenize(doc))
        sent_seqs = []
        for sent in sent_tokenize(doc):
            sent_seqs.append(my_tokenize(sent))
        doc_word_sseqs.append(sent_seqs)  # [[[w11, w12, ...], [w21, w22, ...], ...], ...]
    # stat and index
    lens = []
    doc_lens = []
    doc_sentlens = []
    doc_wordlens = []
    for word_seq in word_seqs:
        counter.update(word_seq)
        lens.append(len(word_seq))
    for word in lexicon.keys():
        counter.update([word])
    for doc_word_seq in doc_word_seqs:
        # counter.update(doc_word_seq)
        doc_lens.append(len(doc_word_seq))
    for sent_seqs in doc_word_sseqs:
        doc_sentlens.append(len(sent_seqs))
        for sent_seq in sent_seqs:
            counter.update(sent_seq)
            doc_wordlens.append(len(sent_seq))
    percentage = 98
    maxlen = int(np.percentile(lens, percentage))
    doc_maxlen_sent = int(np.percentile(doc_sentlens, percentage))  # max sent per doc
    doc_maxlen_word = int(np.percentile(doc_wordlens, percentage))  # max word per sent
    doc_maxlen_word = max(maxlen, doc_maxlen_word)

    # the vocabulary
    min_freq = 3
    word2index = dict()
    idx = 2  # start from 2, 0 as <PAD>, 1 as <OOV>
    for word_count in counter.most_common():
        if word_count[1] >= min_freq or word_count[0] in lexicon:
            word2index[word_count[0]] = idx
            idx += 1
    n_words = len(word2index) + 2
    print('words:', len(word2index))

    print('[agree] words:')
    for word in agree_words:
        if word in word2index:
            agree_indices.add(word2index[word])
            print(' -', word, word2index[word])
    print('[disagree] words:')
    for word in disagree_words:
        if word in word2index:
            disagree_indices.add(word2index[word])
            print(' -', word, word2index[word])
    print('agree: {}\ndisagree: {}'.format(agree_indices, disagree_indices))

    # %% data encoding ====================================================================
    # sent data, and CV version
    seqs = []
    for words in word_seqs:
        seqs.append([word2index.get(word, 1) for word in words])
    padded_seqs_bak = pad_sequences(seqs, maxlen=doc_maxlen_word, padding='post', truncating='post')
    labels_bak = np.asarray(labels, dtype=int)
    print('sent:', padded_seqs_bak.shape, labels_bak.shape)

    # CV-fold split for sentence data
    kf = StratifiedKFold(n_splits=CV, shuffle=True)
    padded_seqs_trains = dict()
    padded_seqs_tests = dict()
    labels_trains = dict()
    labels_tests = dict()
    print('{} fold train/test splitting'.format(CV))
    for cv, (train_idx, test_idx) in enumerate(kf.split(padded_seqs_bak, labels_bak)):
        padded_seqs_trains[cv] = padded_seqs_bak[train_idx]
        padded_seqs_tests[cv] = padded_seqs_bak[test_idx]
        labels_trains[cv] = labels_bak[train_idx]
        labels_tests[cv] = labels_bak[test_idx]

    # lex data
    lex_seqs = []
    for word in lex_word_seqs:
        lex_seqs.append([word2index.get(word, 1)])
    lex_padded_seqs = pad_sequences(lex_seqs, maxlen=1, padding='post', truncating='post')
    lex_labels = np.asarray(lex_labels, dtype=int)
    print(' - lex (all):', lex_padded_seqs.shape, lex_labels.shape)

    # doc data (hierarchical), padding from word to sent
    n_samples = len(doc_word_sseqs)
    doc_padded_seqs = np.zeros(shape=(n_samples, doc_maxlen_sent, doc_maxlen_word), dtype=int)
    for i, sseq_1doc in enumerate(doc_word_sseqs):
        for j, seq_1doc in enumerate(sseq_1doc):
            if j < doc_maxlen_sent:
                for k, word in enumerate(seq_1doc):
                    if k < doc_maxlen_word:
                        doc_padded_seqs[i, j, k] = word2index.get(word, 1)
    doc_labels = np.asarray(doc_labels, dtype=int)
    print(' - doc (all):', doc_padded_seqs.shape, doc_labels.shape)

    # relation data for doc (internal sents) (agree & disagree)
    count_agree, count_disagree = 0, 0
    doc_rel_padded_seqs = np.zeros(shape=(n_samples, doc_maxlen_sent), dtype=int)
    for i in range(0, n_samples):
        for j in range(1, doc_maxlen_sent):
            if doc_padded_seqs[i, j, 0] in agree_indices:
                doc_rel_padded_seqs[i, j] = 1
                count_agree += 1
            if doc_padded_seqs[i, j, 0] in disagree_indices:
                doc_rel_padded_seqs[i, j] = -1
                count_disagree += 1
    print(' - doc sent-rel (all):', doc_rel_padded_seqs.shape)
    print(' - doc sent-rel (all): agree: {}, disagree: {}'.format(count_agree, count_disagree))

    ## sub-sample from lexicon and documents
    print('sub-sampling:')
    # doc data sub-sample
    n_samples = len(padded_seqs_trains[0]) + len(padded_seqs_tests[0])
    doc_padded_seqs, doc_rel_padded_seqs, doc_labels = balanced_subsample3(
        doc_padded_seqs, doc_rel_padded_seqs, doc_labels, subsample_num=n_samples)
    doc_padded_seqs = np.asarray(doc_padded_seqs)
    doc_labels = np.asarray(doc_labels, dtype=int)
    print(' - doc (sampled):', doc_padded_seqs.shape, doc_labels.shape)

    # lex data sub-sample
    lex_padded_seqs, lex_labels = balanced_subsample2(lex_padded_seqs, lex_labels, subsample_num=n_samples)
    lex_padded_seqs = np.asarray(lex_padded_seqs)
    lex_labels = np.asarray(lex_labels, dtype=int)
    print(' - lex (sampled):', lex_padded_seqs.shape, lex_labels.shape)
    ddata = {
        'n_samples': n_samples,
        'n_words': n_words,
        'doc_maxlen_word': doc_maxlen_word,
        'doc_maxlen_sent': doc_maxlen_sent,
        'word2index': word2index,
        'padded_seqs_trains': padded_seqs_trains,
        'labels_trains': labels_trains,
        'padded_seqs_tests': padded_seqs_tests,
        'labels_tests': labels_tests,
        'lex_padded_seqs': lex_padded_seqs,
        'lex_labels': lex_labels,
        'doc_padded_seqs': doc_padded_seqs,
        'doc_labels': doc_labels,
        'doc_rel_padded_seqs': doc_rel_padded_seqs,
    }
    return ddata


def get_model(args, ddata):
    word2index = ddata['word2index']
    n_words = ddata['n_words']
    doc_maxlen_word = ddata['doc_maxlen_word']
    doc_maxlen_sent = ddata['doc_maxlen_sent']

    # word vectors
    print('loading word embeddings')
    embeddings = load_word2vec_bin(vocab=word2index.keys())
    print('processing embedding matrix')
    embedding_mat = get_embedding_mat(embeddings, word2index, args.embed_dim, idx_from=2)
    weights = [embedding_mat]

    ## build model
    # common embeddings
    this_weights = deepcopy(weights)
    embedding_layer = layers.Embedding(input_dim=n_words, output_dim=args.embed_dim, weights=this_weights, name='embeddings')

    # common sentiment classifier
    csc_dense1 = layers.Dense(units=args.hidden_dim, activation='relu', name='csc_dense')
    csc_dropout = layers.Dropout(rate=args.dropout, name='csc_dropout')
    csc_dense2 = layers.Dense(units=1, activation='sigmoid', name='output')
    csc_layers = [csc_dense1, csc_dropout, csc_dense2]  # (*, hidden_dim) -> (*, 1)

    # sent part & sent encoder (to be time-distributed)
    sent_inputs = layers.Input(shape=(doc_maxlen_word,), name='sent_inputs')  # (*, doc_maxlenword)
    sent_embeddings = embedding_layer(sent_inputs)  # (*, doc_maxlenword, embed_dim)
    sent_embeddings = layers.SpatialDropout1D(rate=args.dropout)(sent_embeddings)  # (*, doc_maxlenword, embed_dim)
    sent_hidden = layers.Bidirectional(
        layers.LSTM(units=args.rnn_dim // 2, return_sequences=True,
                    dropout=args.rnn_dropout))(sent_embeddings)  # (*, rnn_dim)
    sent_hidden = layers.GlobalMaxPool1D()(sent_hidden)
    sent_hidden = layers.Dense(args.hidden_dim, activation='relu')(sent_hidden)
    sent_hidden = layers.Dropout(args.dropout)(sent_hidden)  # (*, hidden_dim)
    sent_preds = sent_hidden
    for csc_layer in csc_layers:
        sent_preds = csc_layer(sent_preds)  # after loop: (*, 1)
    # use intermediate hidden representation as encoder output
    sent_encoder = Model(inputs=sent_inputs, outputs=sent_hidden)  # (*, doc_maxlenword) -> (*, hidden_dim)

    # lex part
    lex_inputs = layers.Input(shape=(1,), name='lex_inputs')  # (*, 1)
    lex_embeddings = embedding_layer(lex_inputs)  # (*, 1, embed_dim)
    lex_embeddings = layers.SpatialDropout1D(rate=args.dropout)(lex_embeddings)  # (*, 1, embed_dim)
    lex_hidden = layers.Flatten()(lex_embeddings)  # (*, embed_dim)
    lex_hidden = layers.Dense(units=args.hidden_dim, activation='relu')(lex_hidden)  # (*, hidden_dim)
    lex_hidden = layers.Dropout(rate=args.dropout)(lex_hidden)  # (*, hidden_dim)
    lex_preds = lex_hidden
    for csc_layer in csc_layers:
        lex_preds = csc_layer(lex_preds)  # after loop: (*, 1)

    ## doc part
    doc_inputs = layers.Input(shape=(doc_maxlen_sent, doc_maxlen_word), name='doc_inputs')  # (*, doc_maxsentlen, doc_maxlenword)
    doc_rel_inputs = layers.Input(shape=(doc_maxlen_sent,), name='doc_rel_inputs')  # (*, doc_maxsentlen)
    doc_sent_outputs = layers.TimeDistributed(sent_encoder, name='doc_sent_outputs')(
        doc_inputs)  # (*, doc_maxsentlen, hidden_dim)

    # (virtual sent predictions, for sent relation use)
    virtual_sent_preds = doc_sent_outputs  # (*, doc_maxsentlen, hidden_dim)
    for idx, csc_layer in enumerate(csc_layers):
        name = 'rels' if idx == len(csc_layers) - 1 else None
        virtual_sent_preds = layers.TimeDistributed(csc_layer, name=name)(virtual_sent_preds)  # (*, doc_maxsentlen, 1)

    # sent -> doc
    doc_hidden = layers.Bidirectional(
        layers.LSTM(units=args.rnn_dim // 2, return_sequences=True,
                    dropout=args.rnn_dropout))(doc_sent_outputs)  # (*, rnn_dim)
    doc_hidden = layers.GlobalMaxPool1D()(doc_hidden)
    doc_hidden = layers.Dense(args.hidden_dim, activation='relu')(doc_hidden)
    doc_hidden = layers.Dropout(args.dropout)(doc_hidden)
    doc_preds = doc_hidden
    for csc_layer in csc_layers:
        doc_preds = csc_layer(doc_preds)  # after loop: (*, 1)

    def virtual_sent_loss(_sent_rels):
        """used for calculating loss from sentence relations"""
        def loss(y_true, y_pred):
            sent_rels = K.expand_dims(_sent_rels, axis=-1)  # (*, doc_maxsentlen, 1)
            virtual_pred1 = K.temporal_padding(y_pred, [1, 0])  # (*, doc_maxsentlen+1, 1)
            virtual_pred2 = K.temporal_padding(y_pred, [0, 1])  # (*, doc_maxsentlen+1, 1)
            virtual_pred_diff = virtual_pred1 - virtual_pred2  # (*, doc_maxsentlen+1, 1)
            virtual_pred_diff = virtual_pred_diff[:, :-1, :]  # (*, doc_maxsentlen, 1)
            virtual_pred_diff = K.squeeze(virtual_pred_diff, axis=-1)  # (*, doc_maxsentlen)
            virtual_pred_diff = K.square(virtual_pred_diff)  # (*, doc_maxsentlen)
            sent_rels = K.squeeze(sent_rels, axis=-1)  # (*, doc_maxsentlen)
            loss_sum = K.batch_dot(K.square(virtual_pred_diff), sent_rels, axes=(1, 1))  # (*, 1)
            loss_value = K.mean(K.squeeze(loss_sum, axis=-1))
            return loss_value
        return loss

    model = Model(inputs=[sent_inputs, lex_inputs, doc_inputs, doc_rel_inputs],
                  outputs=[sent_preds, lex_preds, doc_preds, virtual_sent_preds])
    model.compile(
        optimizer='adadelta',  # 'adadelta'
        loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', virtual_sent_loss(doc_rel_inputs)],
        metrics=['acc'],
        loss_weights=[1., args.clex, args.cdoc, args.crel]
    )
    return model


def train_model(args, ddata, gen_model_func):
    n_samples = ddata['n_samples']

    padded_seqs_trains = ddata['padded_seqs_trains']
    padded_seqs_tests = ddata['padded_seqs_tests']
    labels_trains = ddata['labels_trains']
    labels_tests = ddata['labels_tests']

    lex_padded_seqs = ddata['lex_padded_seqs']
    lex_labels = ddata['lex_labels']
    doc_padded_seqs = ddata['doc_padded_seqs']
    doc_labels = ddata['doc_labels']
    doc_rel_padded_seqs = ddata['doc_rel_padded_seqs']

    #--------------------------- CV runs ---------------------------
    all_scores = []
    all_f1s = []
    for run_idx in range(CV):
        print('\n===== run for {}/{} ====='.format(run_idx + 1, CV))
        ## shuffle doc & lex data for this run
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        lex_padded_seqs = lex_padded_seqs[indices]
        lex_labels = lex_labels[indices]
        doc_padded_seqs = doc_padded_seqs[indices]
        doc_labels = doc_labels[indices]

        ## split
        print('split:')
        split_at = len(padded_seqs_trains[run_idx])
        # sent (uses cv split)
        X_train, X_test = padded_seqs_trains[run_idx], padded_seqs_tests[run_idx]
        y_train, y_test = labels_trains[run_idx], labels_tests[run_idx]
        print(' - train:', X_train.shape, y_train.shape)
        print(' - test:', X_test.shape, y_test.shape)
        # doc
        doc_X_train, doc_X_test = doc_padded_seqs[:split_at], doc_padded_seqs[split_at:]
        doc_rel_X_train, doc_rel_X_test = doc_rel_padded_seqs[:split_at], doc_rel_padded_seqs[split_at:]
        doc_y_train, doc_y_test = doc_labels[:split_at], doc_labels[split_at:]
        print(' - doc_train:', doc_X_train.shape, doc_y_train.shape)
        print(' - doc_test:', doc_X_test.shape, doc_y_test.shape)
        # lex
        lex_X_train, lex_X_test = lex_padded_seqs[:split_at], lex_padded_seqs[split_at:]
        lex_y_train, lex_y_test = lex_labels[:split_at], lex_labels[split_at:]
        print(' - lex_train:', lex_X_train.shape, lex_y_train.shape)
        print(' - lex_test:', lex_X_test.shape, lex_y_test.shape)

        ## callbacks
        reducer = callbacks.ReduceLROnPlateau(factor=0.1, patience=args.reduce_patience, verbose=1)
        stopper = callbacks.EarlyStopping(patience=args.stop_patience, verbose=1)
        cbks = [reducer, stopper]

        model = gen_model_func(args=args, ddata=ddata)

        model.fit(
            [X_train, lex_X_train, doc_X_train, doc_rel_X_train],
            [y_train, lex_y_train, doc_y_train, np.zeros((doc_X_train.shape[:2] + (1,)))],
            batch_size=args.batch_size,
            epochs=args.max_epochs,
            verbose=2,
            callbacks=cbks,
            validation_split=0.1,
            shuffle=True,
        )

        score = model.evaluate([X_test, lex_X_test, doc_X_test, doc_rel_X_test],
                               [y_test, lex_y_test, doc_y_test, np.zeros((doc_X_test.shape[:2] + (1,)))],
                               verbose=0)

        y_test_pred = model.predict(x=[X_test, lex_X_test, doc_X_test, doc_rel_X_test], verbose=0)
        y_test_pred = y_test_pred[0]
        y_test_pred = y_test_pred.flatten()  # (*, 1) -> (*,)
        y_test_pred = np.asarray(y_test_pred >= 0.5, dtype=np.int)  # cast to labels

        report = metrics.classification_report(y_true=y_test, y_pred=y_test_pred, target_names=['-', '+'], digits=4)
        print(report)

        _, _, f1, _ = metrics.precision_recall_fscore_support(y_test, y_test_pred, average='macro')
        all_f1s.append(f1)

        # add to total
        all_scores.append(score)

        print('\nCurrent statistics:')
        all_scores_avg = np.mean(np.asarray(all_scores), axis=0)
        acc_avg = all_scores_avg[-4]
        print('- avg acc: {:.4f}'.format(acc_avg))
        print('- avg mf1: {:.4f}'.format(np.mean(all_f1s)))
        print('\nthe end of run {}/{}\n'.format(run_idx + 1, CV))


if __name__ == '__main__':
    pass
