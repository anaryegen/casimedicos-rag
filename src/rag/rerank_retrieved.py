import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Salesforce/SFR-Embedding-2_R")

with open('/gaueko0/users/ayeginbergenov001/MedExpQA-edited/bm25_pubmed_rag.txt') as f:
    data = json.load(f) 




def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a medical query, retrieve the most relevant passages that answer the query'
# queries = [
#     get_detailed_instruct(task, 'How to bake a chocolate cake'),
#     get_detailed_instruct(task, 'Symptoms of the flu')
# ]
# # No need to add instruction for retrieval documents
# passages = [
#     "To bake a delicious chocolate cake, you'll need the following ingredients: all-purpose flour, sugar, cocoa powder, baking powder, baking soda, salt, eggs, milk, vegetable oil, and vanilla extract. Start by preheating your oven to 350°F (175°C). In a mixing bowl, combine the dry ingredients (flour, sugar, cocoa powder, baking powder, baking soda, and salt). In a separate bowl, whisk together the wet ingredients (eggs, milk, vegetable oil, and vanilla extract). Gradually add the wet mixture to the dry ingredients, stirring until well combined. Pour the batter into a greased cake pan and bake for 30-35 minutes. Let it cool before frosting with your favorite chocolate frosting. Enjoy your homemade chocolate cake!",
#     "The flu, or influenza, is an illness caused by influenza viruses. Common symptoms of the flu include a high fever, chills, cough, sore throat, runny or stuffy nose, body aches, headache, fatigue, and sometimes nausea and vomiting. These symptoms can come on suddenly and are usually more severe than the common cold. It's important to get plenty of rest, stay hydrated, and consult a healthcare professional if you suspect you have the flu. In some cases, antiviral medications can help alleviate symptoms and reduce the duration of the illness."
# ]

scores = []
data_keys = list(data.keys())
for key in data_keys:
    task = 'Given a medical querstion, retrieve the most relevant passages that answer the question'
    query = get_detailed_instruct(task, data[key]['question'])
    passages = data[key]['bm25']
    break

# for candidate in passages:
embeddings = model.encode([query] + passages)
scores.append(model.similarity(embeddings[:1], embeddings[1:]) * 100)
    # print(candidate)
    # print('\n\n')
    # print(query)
    # print('\n\n')
    # print(scores)
    # break
print(scores)
print(max(scores[0].tolist()[0]))
print(passages[scores[0].tolist()[0].index(max(scores[0].tolist()[0]))])

