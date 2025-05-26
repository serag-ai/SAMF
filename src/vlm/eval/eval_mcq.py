import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from vlm.src.dataset.multi_dataset import (
    CT_RATE_Multi_Choice_Dataset,
)
from vlm.src.model.language_model import LlavaPhi3ForCausalLM


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./weights/hf",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument(
        "--data_root",
        type=str,
        default="PATH_TO_TEST_DIR/",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="PATH_TO_TEST_CSV_FILE.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_vqa",
    )

    parser.add_argument("--proj_out_num", type=int, default=256)

    return parser.parse_args(args)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    model = LlavaPhi3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        cache_dir=None,
    )

    model = model.to(device=device)

    model.eval()

    test_dataset = CT_RATE_Multi_Choice_Dataset(
        args,
        tokenizer=tokenizer,
        csv_cap_path=args.csv_file,
        mode="test",
    )  # test1k

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=32,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, "eval_mcq.csv")
    with open(output_path, mode="w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            [
                "Question",
                "Ground Truth",
                "pred",
            ]
        )
        with torch.no_grad():
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                answer = sample["answer"]

                input_id = tokenizer(question, return_tensors="pt")["input_ids"].to(
                    device=device
                )
                image = sample["image"].to(device=device)

                generation = model.generate(
                    image,
                    input_id,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                generated_texts = tokenizer.batch_decode(
                    generation, skip_special_tokens=True
                )

                writer.writerow(
                    [
                        question[0],
                        answer[0],
                        generated_texts[0],
                    ]
                )


if __name__ == "__main__":
    main()
