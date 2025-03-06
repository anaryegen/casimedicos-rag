from rank_bm25 import BM25Okapi
import nltk
# nltk.download('punkt')
from datasets import load_dataset

class BM25DocumentRetriever:
    def __init__(self, document_text):
        # self.document_path = document_path
        self.document_text = document_text
        self.sentences = self._preprocess_text(self.document_text)
        self.bm25 = BM25Okapi([sentence.split() for sentence in self.sentences])

    # def _fetch_document_text(self):
    #     with open(self.document_path, 'r', encoding='utf-8') as file:
    #         return file.read()

    def _preprocess_text(self, text):
        return nltk.sent_tokenize(text)

    def retrieve(self, query, top_k=5):
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.sentences[i] for i in top_k_indices]

# Example usage
if __name__ == "__main__":
    # document_path = "/proiektuak/antidote/medical_mt5_corpora/en_es_fr_it.corpus.txt"
    documents = load_dataset('ccdv/pubmed-summarization')
    documents = '\n'.join([text.strip() for text in documents['train']['article']])
    retriever = BM25DocumentRetriever(documents)
    query = "After a traffic accident a 38-year-old patient is admitted to the ICU in coma. After several days the patient does not improve neurologically and a CT scan shows hemorrhagic punctate lesions in the corpus callosum and cortico-subcortical junction. What is the diagnosis?"
    top_k_sentences = retriever.retrieve(query, top_k=5)
    for sentence in top_k_sentences:
        print(sentence)