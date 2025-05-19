# [MICCAI' 25] From Slices to Volumes: Multi-Scale Fusion of 2D and 3D Features for CT Scan Report Generation

This is an official implementation of **[MICCAI 2025]** - From Slices to Volumes: Multi-Scale Fusion of 2D and 3D Features for CT Scan Report Generation

## Workflow overview
 <p align="center">
  <img align="center" src="assets/arch.svg" width="500px"/>
 </p>

A methodology that utilize both the high-level spatial features of 3D data and the rich local details of 2D slices. Our approach begins by pretraining a 2D encoder using a self-supervised learning framework on CT scan slices from three planes: axial, coronal, and sagittal. The outputs of this 2D encoder are then processed by a 3D aggregator to preserve volumetric temporal relationships between slices. Additionally, we introduce a novel fusion technique that integrates the outputs of the aggregator, the 2D encoder, and a prompt, effectively bridging the gap between 2D and 3D representations. This fused representation is then fed into an LLM to generate medical reports.

## Finetuning

To fine-tune the model, run following step:

**Run Fine-Tuning Script** 🛠️:
   ```sh
   sh script/finetune_phi3.sh
   ```

## Merge

After finetuning, you need to merge LoRA weights with the original weights. Follow :

**Run the Merge Script** 🔄:
   ```sh
   python3 -u merge_lora_weights_and_save_hf_model.py \
   --model_type phi3 \
   --model_with_lora PATH_TO_FINETUNED_MODEL \
   --mm_fuse_type samf \
   --output_dir PATH_TO_OUTPUT_DIR/
   ```

## Evaluation

To perform evaluation using provided metrics, follow :

**Run the vlm/eval/eval_caption.py Script** 🔄:
   ```sh
    python3 -u /src/vlm/eval/eval_caption.py \
    --model_name_or_path PATH_TO_MERGED_MODEL/ \
    --output_dir PATH_TO_OUTPUT_DIR
   ```

## Dataset

Downlaod [CT-Rate](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) dataset. For this work, we have converted the data into CSV files. The format for training captions should be as follows:

| **image**   | **caption** | **label** |
|-------------|-------------|-----------|
| image_path  | text        | organ     |



## Acknowledgement
We appreciate open source projects including: 
[LLaVA](https://github.com/haotian-liu/LLaVA) and 
[M3d](https://github.com/BAAI-DCAI/M3D), 