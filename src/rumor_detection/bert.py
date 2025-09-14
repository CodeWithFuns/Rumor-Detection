import argparse, os, json
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from rumor_detection.data import load_dataset

# Optional heavy deps
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch
from datasets import Dataset as HFDataset

@dataclass
class Args:
    data: str
    text_col: str
    label_col: str
    epochs: int
    batch_size: int
    save_dir: str
    model_name: str

def encode_labels(y):
    classes = sorted(list(set(y)))
    to_id = {c:i for i,c in enumerate(classes)}
    ids = [to_id[v] for v in y]
    return np.array(ids), classes

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=str)
    ap.add_argument("--text-col", default="text", type=str)
    ap.add_argument("--label-col", default="label", type=str)
    ap.add_argument("--epochs", default=1, type=int)
    ap.add_argument("--batch-size", default=8, type=int)
    ap.add_argument("--save-dir", default="models", type=str)
    ap.add_argument("--model-name", default="distilbert-base-uncased", type=str)
    return ap.parse_args()

def main():
    args = parse_args()
    X, y = load_dataset(args.data, args.text_col, args.label_col)
    y_ids, classes = encode_labels(y)

    x_train, x_val, y_train, y_val = train_test_split(X, y_ids, test_size=0.2, random_state=42, stratify=y_ids)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize(batch):
        return tok(list(batch["text"]), truncation=True, padding="max_length", max_length=256)

    ds_train = HFDataset.from_dict({"text": list(x_train), "labels": list(y_train)}).map(tokenize, batched=True)
    ds_val   = HFDataset.from_dict({"text": list(x_val), "labels": list(y_val)}).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(classes))

    training_args = TrainingArguments(
        output_dir="models/bert_out",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        from sklearn.metrics import accuracy_score, f1_score
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Save
    save_path = os.path.join(args.save_dir, "bert_distilbert")
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    with open(os.path.join(save_path, "label_mapping.json"), "w") as f:
        json.dump({"classes": classes}, f, indent=2)

    # Write simple metrics file
    metrics_path = os.path.join("reports", "metrics_bert.json")
    os.makedirs("reports", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print("Saved BERT model to:", save_path)

if __name__ == "__main__":
    main()
