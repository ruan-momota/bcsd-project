import os
import json
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import config


class BCSDTripletDataset(Dataset):
    """
    Triplet dataset for BCSD training.

    Each sample returns:
        anchor, positive (same function)
        negative (different function)
    """

    def __init__(self, projects, epoch_sample_rate=15, blocklist_file=None):

        self.projects = projects
        self.epoch_sample_rate = epoch_sample_rate

        self.base_dir = os.path.join(
            config.DATA_DIR,
            "outputs",
            "student",
            "256_5",
        )

        if blocklist_file is None:
            blocklist_file = os.path.join(
                config.DATA_DIR,
                "outputs",
                "blocklist256_5.json",
            )

        # storage
        self.samples = []
        self.groups = defaultdict(list)  # func_name -> indices in self.samples

        # -----------------------------
        # Load blocklist
        # -----------------------------
        self.blocklist = set()

        if os.path.exists(blocklist_file):
            with open(blocklist_file, "r", encoding="utf-8") as f:
                self.blocklist = set(json.load(f))
        else:
            print(
                f"[Warning] Blocklist file not found at {blocklist_file}. "
                "Proceeding without filtering."
            )

        print(f"Loading pre-tokenized data for projects: {self.projects}")

        total_files = 0
        skipped_count = 0

        # -----------------------------
        # Load dataset
        # -----------------------------
        for proj_name in tqdm(self.projects, desc="Loading Projects"):

            proj_path = os.path.join(self.base_dir, proj_name)

            if not os.path.exists(proj_path):
                print(
                    f"[Warning] Project directory not found: {proj_path}. Skipping."
                )
                continue

            pt_files = sorted(
                [f for f in os.listdir(proj_path) if f.endswith(".pt")]
            )

            if not pt_files:
                print(f"[Warning] No .pt files found in {proj_path}.")
                continue

            for pt_file in pt_files:

                file_path = os.path.join(proj_path, pt_file)

                try:
                    chunk_data = torch.load(file_path)
                    clean_chunk = []

                    # -----------------------------
                    # Filter blocklist
                    # -----------------------------
                    for item in chunk_data:

                        unique_key = (
                            f"{item['proj_name']}|"
                            f"{item['file_name']}|"
                            f"{item['func_name']}"
                        )

                        if unique_key in self.blocklist:
                            skipped_count += 1
                            continue

                        clean_chunk.append(item)

                    if not clean_chunk:
                        continue

                    start_idx = len(self.samples)
                    self.samples.extend(clean_chunk)

                    # build func_name groups
                    for i, item in enumerate(clean_chunk):

                        global_idx = start_idx + i
                        func_name = item["func_name"]

                        self.groups[func_name].append(global_idx)

                    total_files += 1

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        # -----------------------------
        # Filter functions with >=2 variants
        # -----------------------------
        self.valid_func_names = [
            k for k, v in self.groups.items() if len(v) >= 2
        ]

        print(f"Processed {total_files} .pt files.")
        print(f"Total samples loaded: {len(self.samples)}")
        print(
            f"Skipped {skipped_count} dirty samples based on blocklist."
        )
        print(
            f"Valid function groups (>=2 variants): "
            f"{len(self.valid_func_names)}"
        )

    # -------------------------------------------------
    # Dataset size
    # -------------------------------------------------
    def __len__(self):
        # number of samples per epoch
        return len(self.valid_func_names) * self.epoch_sample_rate

    # -------------------------------------------------
    # Triplet sampling
    # -------------------------------------------------
    def __getitem__(self, idx):

        group_idx = idx % len(self.valid_func_names)
        group_name = self.valid_func_names[group_idx]

        group_indices = self.groups[group_name]

        if len(group_indices) < 2:
            raise ValueError(
                f"Function '{group_name}' has less than 2 samples."
            )

        # anchor + positive
        anc_idx, pos_idx = random.sample(group_indices, 2)

        # negative
        while True:
            neg_name = random.choice(self.valid_func_names)

            if neg_name != group_name:
                neg_indices = self.groups[neg_name]
                neg_idx = random.choice(neg_indices)
                break

        item_anc = self.samples[anc_idx]["student_input"]
        item_pos = self.samples[pos_idx]["student_input"]
        item_neg = self.samples[neg_idx]["student_input"]

        # -----------------------------
        # Build tensors
        # -----------------------------
        input_ids = torch.stack(
            [
                item_anc["input_ids"],
                item_pos["input_ids"],
                item_neg["input_ids"],
            ]
        )

        attention_mask = torch.stack(
            [
                item_anc["attention_mask"],
                item_pos["attention_mask"],
                item_neg["attention_mask"],
            ]
        )

        token_type_ids = torch.stack(
            [
                item_anc["token_type_ids"],
                item_pos["token_type_ids"],
                item_neg["token_type_ids"],
            ]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }