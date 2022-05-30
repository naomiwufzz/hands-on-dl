def load_reuters():
    from nltk.corpus import reuters
    text = reuters.sents()
    text = [[word.lower() for word in sentence] for sentence in text]
    vocab = Vocab.build()
