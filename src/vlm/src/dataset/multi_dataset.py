import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import transforms

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import set_track_meta
from PIL import Image
from .prompt_templates import (
    Caption_templates,
)
import nibabel as nib


class CT_RATE_CapDataset(Dataset):
    def __init__(self, args, csv_path, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        self.data_list = pd.read_csv(csv_path)

        self.caption_prompts = Caption_templates

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        set_track_meta(False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)
                nii_image = nib.load(image_abs_path)
                image = nii_image.get_fdata()

                # GET SLICES
                z_dim = image.shape[2]
                mid_index = z_dim // 2
                margin = 3 * 24
                # Calculate the range of indices
                start_index = mid_index - margin
                end_index = mid_index + margin
                # Ensure the indices stay within bounds
                start_index = max(0, start_index)
                end_index = min(z_dim - 1, end_index)
                indices = np.linspace(start_index, end_index, 24, dtype=int)

                slices = [image[:, :, i] for i in indices]

                image = torch.stack(
                    [
                        self.transform(Image.fromarray(slice_.astype(np.uint8)))
                        for slice_ in slices
                    ]
                )

                answer = data["caption"]

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "Caption",
                }

                return ret

            except Exception as e:
                print(
                    f"Error in __getitem__ at index {os.path.join(self.data_root, self.data_list.iloc[idx]['image'])}: {e}"
                )
                idx = random.randint(0, len(self.data_list) - 1)


class CT_RATE_Multi_Choice_Dataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        csv_cap_path="./dataset/sample_mcq_data.csv",
        close_ended=True,
        mode="train",
    ):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<im_patch>" * args.proj_out_num

        self.data_list = pd.read_csv(csv_cap_path)

        self.images_path = self.data_list["Image"]
        self.questions = self.data_list["Question"]
        self.answers = self.data_list["Answer"]
        self.answer_choices = self.data_list["Answer Choice"]
        self.choices_a = self.data_list["Choice A"]
        self.choices_b = self.data_list["Choice B"]
        self.choices_c = self.data_list["Choice C"]
        self.choices_d = self.data_list["Choice D"]

        self.caption_prompts = Caption_templates

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        set_track_meta(False)

    def __len__(self):
        return len(self.data_list)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                image_path = self.images_path[idx]
                image_abs_path = os.path.join(self.data_root, image_path)
                nii_image = nib.load(image_abs_path)
                image = nii_image.get_fdata()

                # GET SLICES
                z_dim = image.shape[2]
                mid_index = z_dim // 2
                margin = 3 * 24
                # Calculate the range of indices
                start_index = mid_index - margin
                end_index = mid_index + margin
                # Ensure the indices stay within bounds
                start_index = max(0, start_index)
                end_index = min(z_dim - 1, end_index)
                indices = np.linspace(start_index, end_index, 24, dtype=int)

                slices = [image[:, :, i] for i in indices]
                # image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = torch.stack(
                    [
                        self.transform(Image.fromarray(slice_.astype(np.uint8)))
                        for slice_ in slices
                    ]
                )

                if self.close_ended:
                    question = self.questions[idx]
                    choices = "Choices: a. {} b. {} c. {} d. {}".format(
                        self.choices_a[idx],
                        self.choices_b[idx],
                        self.choices_c[idx],
                        self.choices_d[idx],
                    )
                    question = question + " " + choices
                    answer = "{}. {}.".format(
                        self.answer_choices[idx], self.answers[idx]
                    )
                else:
                    question = self.questions[idx]
                    answer = str(self.answers[idx])

                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": self.answer_choices[idx],
                    "question_type": "Caption",
                }

                return ret
            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, self.__len__() - 1)


class CT_RATE_UniDatasets(Dataset):
    def __init__(self, args, cap_csv_path, mcq_csv_path, tokenizer, mode="train"):
        super(CT_RATE_UniDatasets, self).__init__()
        self.ds_list = [
            CT_RATE_CapDataset(
                args, csv_path=cap_csv_path, tokenizer=tokenizer, mode=mode
            ),
            CT_RATE_Multi_Choice_Dataset(
                args, tokenizer, csv_cap_path=mcq_csv_path, close_ended=True, mode=mode
            ),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
