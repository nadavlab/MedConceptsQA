Install the required dependencies:
```
pip install -r requirements.txt
```

Run the benchmark evaluation:
```
python evaluation_runner.py --model_id MODEL_ID --output_results_dir_path OUTPUT_RESULTS_DIR_PATH
```
Replace `MODEL_ID` with the model name (HuggingFace) or local path to the pretrained model you want to evaluate.

Replace `OUTPUT_RESULTS_DIR_PATH` with the directory path to store the results CSV file.

