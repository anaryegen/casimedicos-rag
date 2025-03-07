# Casimedicos-rag

To launch run the `.sh` script with the followiing command: 
```
for SHOT in 0 1 2 3 4 5
do
for LANG in en es it fr
do
	python3 ./src/run.py --shot $SHOT --config_file configs/$LANG/zero_shot/Qwen2.5_7B_none.yaml
done
done
```
Experiments: 

- `Q(uestion) + O(ptions)`

    

| Experiments | Mode| Status |    
|-------------|---  |-----   |
| zero-shot   |  Q + O   | ✅     | 
|few-shot     |  Q + O   | ✅     | 
|fine-tuning  |  Q + O   | ✅     | 
| multilingual RAG | Q + D + O | ⌛️     |    

*D == Document

- `Question + ~CoT/Explanations + O`

| Experiments | Mode| Status |    
|-------------|---  |-----   |
| zero-shot   |  Q + CoT + O   |     | 
|few-shot     |  Q + CoT + O   |     | 
|fine-tuning  |  Q + CoT + O   |     | 

<!-- ## Results [link](https://docs.google.com/spreadsheets/d/1c7Po3dkRZWfrvEoRnxXEsr6B4zZ_Hgut8QTTsN89kf4/edit?gid=0#gid=0) -->
<!-- ## Colab [link](https://colab.research.google.com/drive/1fyVEt178rkd_ifP1SxD2j75N4C-unawL#scrollTo=GQZK1PbRbWN7) with potential EDA   -->


<!-- | multilingual RAG | Q + D + O |     |  -->

<!-- # MedExpQA
This repository contains code for [MedExpQA: Multilingual Benchmarking of Large Language
Models for Medical Question Answering]().

We release all model LoRA adapter checkpoints, as well as the datasets and code to train and evaluate them. This 
repository also contains the code to augment the dataset with the retrieved data augmentation.

## Getting Started
Clone this GitHub repository, install the requirements, and download all [datasets](https://huggingface.co/datasets/HiTZ/MedExpQA) and [model LoRA adapter checkpoints](). 
This project was developed using **Python=3.9.18**. 

```
git clone https://github.com/hitz-zentroa/MedExpQA.git
cd MedExpQA
pip install -r requirements.txt
```

## Datasets
Download the datasets [here](https://huggingface.co/datasets/HiTZ/MedExpQA) and place the `.jsonl` files in `./data/casimedicos/`.

<!--- ## Model checkpoints -->
<!--- Download model LoRA adapter checkpoints [here]() and place each model's folder in `./out/experiments/finetuned/`. -->
<!-- 
## Configuration codenames
These are the internal codenames for grounding configurations:
- **None** `none`
- **Full gold explanation (E):** `full`
- **Gold Explanations of the Incorrect Options (EI):** `other`
- **Full gold explanation with Hidden explicit references to the correct/incorrect answer (H):** `clean`
- **RAG with up to 7 grounding snippets (RAG-7):** `ragcc`
- **RAG with up to 32 grounding snippets (RAG-32):** `ragccmax`

## Training models
To train each of the featured models run `./src/run.py` and point at the configuration you want to execute the training 
with. Different configuration files can be found in the `configs` folder. For example, launching a 5 epoch fine-tuning of 
BioMistral (7b) using RAG-7 (RAG with up to 7 grounding snippets) run:
```
export PYTHONPATH="$PWD/src"
LANG="en" # Langue of the CasiMedicos dataset. Can be [en | es | fr | it]
python3 ./src/run.py configs/grounded/classification/$LANG/zero_shot/BioMistral_7b_ragcc_en.yaml
```
Inference on the test set for each checkpoint will be performed and resulting predictions will be stored in the 
`output_dir` folder set in the configuration file. 

## Performing inference
You can use one of the fine-tuning configurations under the `fine_tuning` config folder. Set `do_train: false` and 
`do_eval: false`. <!--- To load the adapter you have two options: -->
<!--- 1. Load the adapter directly from HuggingFace and adding the `lora_weights_name_or_path` parameter to the configuration you want to launch. For example: `lora_weights_name_or_path: HiTZ/MedExpQA/Mistral-7b-rag-max-EN` -->
<!--- 2. Download the [model LoRA adapter checkpoints]() and leave the checkpoints in `out/experiments/finetuned/`. -->
<!-- Inferences are launched in the same way as trainings:
```
export PYTHONPATH="$PWD/src"
LANG="en" # Langue of the CasiMedicos dataset. Can be [en | es | fr | it]
python3 ./src/run.py configs/grounded/classification/$LANG/zero_shot/Mistral_7b_ragccmax_en.yaml
```
The resulting predictions will be stored in the `output_dir` folder set in the configuration file.

## Evaluating predictions
Write the paths to the folders were the prediction files are stored on a file (an example can be found in 
`configs/predictions_to_eval.txt`) and pass this file as an argument of `evaluate_predictions.py`. Example:
```
export PYTHONPATH="$PWD/src"
python3 ./src/model/casimedicosmt5/evaluate_predictions.py configs/predictions_to_eval.txt
``` --> 
