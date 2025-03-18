from rank_bm25 import BM25Okapi
import nltk
# nltk.download('punkt')
from datasets import load_dataset
from glob import glob
import json
from collections import defaultdict 
from tqdm import tqdm
#TODO
# - refactor to other files to make it more flexible to changes woth different modes of retrieval

# for reranking
# Salesforce/SFR-Embedding-2_R

class BM25DocumentRetriever:
    def __init__(self, document_text):
        # self.document_path = document_path
        self.document_text = document_text
        self.document_text = document_text.split('\n')
        #self.sentences = self._preprocess_text(self.document_text)
        self.bm25 = BM25Okapi([text.split() for text in self.document_text])

    # def _fetch_document_text(self):
    #     with open(self.document_path, 'r', encoding='utf-8') as file:
    #         return file.read()

    def _preprocess_text(self, text):
        return nltk.sent_tokenize(text)

    def retrieve(self, query, top_k=5):
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.document_text[i] for i in top_k_indices]

# Example usage
if __name__ == "__main__":
    # document_path = "/proiektuak/antidote/medical_mt5_corpora/en_es_fr_it.corpus.txt"
    # documents = load_dataset('ccdv/pubmed-summarization')
    # documents = '\n'.join([text.strip() for text in documents['train']['article']])
    data = load_dataset('HiTZ/MedExpQA', 'en', split='test').to_pandas()
    questions = data['full_question'].tolist()
    questions_ids = data['id'].tolist()
    options = data['options'].tolist()
    correct_answers = data['correct_option'].tolist()
    # PubMed
    data_files = glob('/proiektuak/antidote/Developer/Python/Projects/antidote/data/rag/pubmed/chunk/*.jsonl')
    retrieved_files_bm25 = open('bm25_pubmed_rag.txt', 'w')

    print(f'There are {len(data_files)} in the retrieval folder')
    final_retrieved = defaultdict(list)
    for i in tqdm(range(0, len(data_files), 100)):  # process every 100 files
        documents = []
        for f in data_files[i:i+100]:
            documents += [eval(l)['content'] for l in open(f).readlines()]
        documents = '\n'.join(documents)
        retriever = BM25DocumentRetriever(documents)
        for j in range(len(questions)):
            question_id = questions_ids[j]
            if question_id not in final_retrieved:
                final_retrieved[question_id] = {
                    'id': question_id,
                    'question': questions[j],
                    'options': options[j],
                    'correct_answer': correct_answers[j],
                    'bm25': []
                }
            top_k_sentences = retriever.retrieve(questions[j], top_k=5)
            final_retrieved[question_id]['bm25'].extend(top_k_sentences)
    # retrieved_files_bm25.write(str(final_retrieved[question_id]) + '\n')
    json.dump(final_retrieved, retrieved_files_bm25, indent=4)
    
    print('Finished!')
