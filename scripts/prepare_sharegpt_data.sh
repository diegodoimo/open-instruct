HF_TOKEN=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
# check if there is $HF_TOKEN in the environment variables
if [ -z "$HF_TOKEN" ]
then
    echo "Warning: HuggingFace dataset LIMA requires permissive access."
    echo "Warning: Please request the access at https://huggingface.co/datasets/GAIR/lima and set the HF_TOKEN environment variable before running this script."
    exit 1
fi


#
#echo "Downloading ShareGPT dataset..."
#wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json
#wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json
#echo "Splitting the ShareGPT dataset with 2048 max tokens per conversation..."
#python scripts/split_sharegpt_conversations.py \
#    --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
#    --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_2048.json \
#    --model-name-or-path /u/area/ddoimo/ddoimo/llama/llama_v2/models_hf/llama-2-7b/ \
#    --max-length 2048
#echo "Splitting the ShareGPT dataset with 4096 max tokens per conversation..."
#python scripts/split_sharegpt_conversations.py \
#    --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
#    --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_4096.json \
#    --model-name-or-path /u/area/ddoimo/ddoimo/llama/llama_v2/models_hf/llama-2-7b/ \
#    --max-length 4096

echo "Processing datasets..."
python open_instruct/reformat_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/
