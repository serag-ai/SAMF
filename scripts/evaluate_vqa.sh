export PYTHONPATH=$PYTHONPATH:/SAMF/src
python3 -u SAMF/src/vlm/eval/eval_mcq.py \
    --model_name_or_path serag-ai/SAMF \
    --output_dir ./evaluation \
    --data_root CHANGE_TO_PATH_DIR_WITH_CTSCAN_IMAGES \
    --csv_file /SAMF/dataset/sample_mcq_data.csv \