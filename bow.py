import numpy as np

class BagOfWord:
    def __init__(self):
        self.vocabulary = {}

    def fit(self, documents):
        words = set()
        for doc in documents:
            words.update(doc.split())

        self.vocabulary = {}
        for idx, word in enumerate(sorted(words)):
            self.vocabulary[word] = idx

    def transform(self, document):
        vector = np.zeros(len(self.vocabulary))
        for word in document.split():
            if word in self.vocabulary:
                vector[self.vocabulary[word]] += 1
        return vector


if __name__ == "__main__":
    docs = [
        "the cat sat on the mat",
        "the dog chased the cat",
        "the mat was sat on by the dog"
    ]
    bow = BagOfWord()
    bow.fit(docs)
    print(f"Vocabulary: {bow.vocabulary}")
    print(f"Vocabulary Size: {len(bow.vocabulary)}")

    text = "the cat the dog"
    bow_vec = bow.transform(text)
    print(f"BoW Vector: {bow_vec}")
