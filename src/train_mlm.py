import os
import random
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertForMaskedLM, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset_mlm import BCSDMLMDataset
from model_eval import evaluate_model
from model import SmallBERT
import config


EPOCHS = 20
BATCH_SIZE = 64
LR = 5e-5
MASK_PROB = 0.15
PAD_TOKEN_ID = 1
MASK_TOKEN_ID = 5
VOCAB_SIZE = 33555
MAX_POSITION_EMBEDDINGS = 256
NUM_WORKERS = 0
SEED = 42
GRAD_CLIP_NORM = 1.0

SAVE_DIR = os.path.join(config.DATA_DIR, "checkpoints", "mlm")
BENCHMARK_DIR = os.path.join(config.DATA_DIR, "bcsd_benchmark_5")
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_mlm(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch])

    labels = input_ids.clone()

    # build masked matrix
    probability_matrix = torch.full(labels.shape, MASK_PROB)
    probability_matrix.masked_fill_(labels == PAD_TOKEN_ID, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 80% -> [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = MASK_TOKEN_ID

    # 10% -> random token
    # among the remaining 20%, choose about half => final ~10%
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=input_ids.shape,
        dtype=torch.long,
    )
    input_ids[indices_random] = random_words[indices_random]

    # unmasked positions should not contribute to loss
    labels[~masked_indices] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


def build_eval_model_from_mlm(mlm_model: BertForMaskedLM, device: torch.device) -> SmallBERT:
    """
    Build a SmallBERT evaluation model and copy/share encoder weights from MLM model.
    """
    eval_model = SmallBERT().to(device)

    # Prefer loading state_dict if architectures are compatible enough.
    # If not, fall back to assigning bert encoder directly.
    try:
        if hasattr(eval_model, "bert") and hasattr(mlm_model, "bert"):
            eval_model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)
        else:
            raise AttributeError("Model has no .bert attribute")
    except Exception:
        # fallback: direct assignment
        eval_model.bert = mlm_model.bert

    return eval_model


def evaluate_safely(eval_model, device, benchmark_dir, mode="val"):
    eval_model.eval()
    with torch.no_grad():
        results = evaluate_model(eval_model, device, benchmark_dir, mode=mode)
    return results


def save_best_eval_model(eval_model: SmallBERT, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)

    # HuggingFace style if supported
    if hasattr(eval_model, "save_pretrained"):
        eval_model.save_pretrained(save_path)
        return

    # fallback
    torch.save(eval_model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))


def load_best_eval_model(save_path: str, device: torch.device) -> SmallBERT:
    # HuggingFace style if supported
    if hasattr(SmallBERT, "from_pretrained"):
        try:
            model = SmallBERT.from_pretrained(save_path)
            return model.to(device)
        except Exception:
            pass

    # fallback
    model = SmallBERT().to(device)
    state_path = os.path.join(save_path, "pytorch_model.bin")
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")

    dataset = BCSDMLMDataset(projects=["openssl", "clamav", "zlib", "nmap"])
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_mlm,
    )

    print("Initializing BertForMaskedLM...")
    bert_config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        type_vocab_size=256,
        pad_token_id=PAD_TOKEN_ID,
    )

    model = BertForMaskedLM(bert_config).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_map = 0.0
    best_model_path = os.path.join(SAVE_DIR, "mlm_model")

    print("Start MLM Training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in loop:
            input_ids = batch["input_ids"].to(device, non_blocking=pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin_memory)
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=pin_memory)
            labels = batch["labels"].to(device, non_blocking=pin_memory)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} Done. Avg MLM Loss: {avg_loss:.4f}")

        # === Evaluation ===
        eval_model = build_eval_model_from_mlm(model, device)
        eval_results = evaluate_safely(eval_model, device, BENCHMARK_DIR, mode="val")

        current_map = eval_results.get("Map@50", 0.0)
        current_pre = eval_results.get("R-Precision", 0.0)
        print(f"Result: Map@50 = {current_map:.4f}, R-Precision = {current_pre:.4f}")

        # Save Best Eval Model
        if current_map > best_map:
            best_map = current_map
            save_best_eval_model(eval_model, best_model_path)
            print(f"New Best MLM-derived Eval Model Saved to {best_model_path}!")

    print("MLM Training Finished.")

    print("\n=== Starting Final Evaluation on Test Set ===")

    try:
        best_model = load_best_eval_model(best_model_path, device)
        print(f"Successfully loaded best model from {best_model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_results = evaluate_safely(best_model, device, BENCHMARK_DIR, mode="test")

    print("\n>>>>>> FINAL TEST RESULTS <<<<<<")
    print(f"Student Model: {best_model_path}")
    for metric_name, score in test_results.items():
        print(f"{metric_name:<15}: {score:.4f}")
    print(">>>>>>>>>>>>>><<<<<<<<<<<<<<")


if __name__ == "__main__":
    main()