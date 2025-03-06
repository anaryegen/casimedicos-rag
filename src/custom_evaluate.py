import json
import argparse
import evaluate 


def eval_predictions(file_path: str):
    accuracy_metric = evaluate.load('accuracy')
    with open(file_path, 'r') as f:
        data = json.load(f)
    preds, trues = [], []
    lang = file_path.split('.')[-5]
    lang_prompt = {
        "es": ["Aquí están las posibles opciones:\n", "La opción correcta es: "],
        "en": ["Here are the potential choices:\n", "The correct answer is: "],
        "it": ["Ecco le opzioni possibili:\n", "L'opzione corretta è: "],
        "fr": ["Voici les options possibles:\n", "La bonne option est: "]
    }
    for d in data:
        if d['prediction']:
            p = d['prediction'].split(lang_prompt[lang][0])[1].split(lang_prompt[lang][1])[1]
            # print("Splitted prediction: ", d['prediction'].split(lang_prompt[lang][0])[1].split(lang_prompt[lang][1])[1][:3])
        else:
            p = [0]
        #print(p)
        #p = [int(i) for i in p if i.isdigit() and i.isnumber()]
        t = d['gold_answer']

        
        p = 0 if not p[0].isdigit() else int(p[0])
        #t = 0 if not t[0].isdigit() else int(t[0])
        t = int(t[0])
        preds.append(p)
        trues.append(t)
        assert len(preds) == len(trues), "Length of predictions and gold answers are not the same. Please make sure you are passing the right data"\

    accuracy = accuracy_metric.compute(references=trues, predictions=preds)

    print(f"\n\nAccuracy for file: {file_path} is {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help="The path to the prediction file which has gold and predicted answers", type=str)
    args = parser.parse_args()
    eval_predictions(args.file_path)
