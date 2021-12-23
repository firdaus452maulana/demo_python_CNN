import csv

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_js_stopword(filename):
    with open(filename, "r") as f:
        data = f.read()
    stopwords = data.split("\n")[-3]
    stopwords = [word[1:-1] for word in stopwords[10:-3].split(", ")]
    # print(stopwords)
    return stopwords


def read_test_text(file_path):
    with open(file_path, "r") as f:
        test_text = f.read()
    return test_text


if __name__ == "__main__":
    stopwords = read_js_stopword("datasets/stopwords.js")
    # Load text
    sentences = []
    label = []
    with open("datasets/bbc-text.csv", "r") as csvfile:
        datastore = csv.reader(csvfile)
    datastore.__next__()  # skip the first line
    for item in datastore:
        text = [word for word in item[1].split() if word not in stopwords]
    sentences.append(" ".join(text))
    label.append(item[0])
    print(f"len(sentences): {len(sentences)}")
    test_sent = read_test_text("datasets/bbc-text-test.txt")
    print(f"sentences[0] == test_sent => {sentences[0] == test_sent}")
    print(f"labels[:10] = {label[:10]}")
    # Fit sentences to create tokens
    sent_tokenizer = Tokenizer(oov_token="<OOV>")
    sent_tokenizer.fit_on_texts(sentences)
    word_index = sent_tokenizer.word_index
    print(f"len(word_index): {len(word_index)}")
    # Tokenizing the sentences
    sequences = sent_tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding="post")
    print(f"padded[0]: {padded[0]}")
    print(f"padded.shape: {padded.shape}")
    # Fit Tokenizer to sentences and create a sequence of token for labels
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(label)
    label_word_index = label_tokenizer.word_index
    label_seq = label_tokenizer.texts_to_sequences(label)
    print(f"label_seq[:10]: {label_seq[:10]}")
    print(f"label_word_index: {label_word_index}")
