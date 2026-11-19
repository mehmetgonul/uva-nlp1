import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.nn.utils.rnn import pad_sequence 
import time
from tqdm.auto import tqdm
from collections import defaultdict, OrderedDict
from pathlib import Path
import itertools
import json

import csv
import gzip
import os
from tempfile import TemporaryDirectory

from . import clslib


class LabelledCorpus(Dataset):
    def __init__(self, X, y):
        """Receives a list of variable-length token-id sequences and labels."""
        self.X = [torch.tensor(x, dtype=torch.long) for x in X]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_labelled_texts(batched_pairs, padding_value=0):
    X_rows, y_rows = zip(*batched_pairs)    
    X_batch = pad_sequence(X_rows, batch_first=True, padding_value=padding_value)
    y_batch = torch.stack(y_rows)
    return X_batch, y_batch


def training_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    if isinstance(device, str):
        device = torch.device(device)
        
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logprobs = model(X_batch)
        loss = criterion(logprobs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        predictions = logprobs.argmax(dim=1)
        total_correct += (predictions == y_batch).sum().item()
        total_examples += y_batch.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }

def evaluate(model, dataloader, criterion, device="cpu"):
    if isinstance(device, str):
        device = torch.device(device)
        
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    preds_list = []
    labels_list = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logprobs = model(X_batch)
            loss = criterion(logprobs, y_batch)

            total_loss += loss.item() * y_batch.size(0)
            predictions = logprobs.argmax(dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_examples += y_batch.size(0)
            preds_list.append(predictions.tolist())
            labels_list.append(y_batch.tolist())

    return {
        "predictions": preds_list,
        "labels": labels_list,
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }

def train(
    training_set, 
    validation_set, 
    tokenizer: clslib.Tokenizer,    
    emb_model: clslib.PretrainedEmbeddings,
    num_classes: int,    
    enc_type: str,
    save_to="temp",    
    enc_kwargs={'freeze_embeddings': True},
    regressor_kwargs={'hidden_size1': 50, 'hidden_size2': 50, 'dropout': 0.},    
    opt_kwargs={'learning_rate': 1e-3, 'weight_decay': 1e-4, 'batch_size': 64, 'num_epochs': 20},    
    device='cpu',
    *,
    seed=None,
    shared_table=None,
):
    """Train on labelled texts, or on token IDs when tokenizer is None."""
    if opt_kwargs['num_epochs'] < 1:
        raise ValueError("num_epochs must be positive")

    time_start = time.time()

    if tokenizer is None:
        train_dataset, val_dataset = training_set, validation_set
    else:
        print("Tokenizing texts...")
        training_vectors = [tokenizer.ids(text) for text, label in training_set]
        training_labels = [int(label) for text, label in training_set]
        validation_vectors = [tokenizer.ids(text) for text, label in validation_set]
        validation_labels = [int(label) for text, label in validation_set]
        train_dataset = LabelledCorpus(training_vectors, training_labels)
        val_dataset = LabelledCorpus(validation_vectors, validation_labels)
        print(f"Tokenization done! Time taken: {time.time() - time_start:.3f} seconds")

    time_tokenizing = time.time()
    pad_id = emb_model.vocab.index(emb_model.pad_token)
    generator = None
    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt_kwargs['batch_size'],
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate_labelled_texts(batch, padding_value=pad_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt_kwargs['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_labelled_texts(batch, padding_value=pad_id),
    )

    time_dataloader = time.time()
    print(f"Dataloader done! Time taken: {time_dataloader - time_tokenizing:.3f} seconds")

    if isinstance(device, str):
        device = torch.device(device)

    # save architecture-defining arguments
    enc_kwargs = dict(enc_kwargs, padding_idx=pad_id)
    regressor_kwargs = dict(regressor_kwargs, num_classes=num_classes)
    args = {        
        'enc_kwargs': enc_kwargs,
        'regressor_kwargs': regressor_kwargs,
        'enc_type': enc_type,
    }
    if shared_table is not None:
        args['embedding_table'] = str(shared_table)
    with open(f"{save_to}.json", "w", encoding="utf-8") as f:        
        json.dump(args, f, indent=2, ensure_ascii=False) 

    if enc_type == 'embed-and-pool':
        encoder = clslib.EmbedAndPoolEncoder(
            embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32),
            **enc_kwargs
         )
    elif enc_type == 'lstm':
         encoder = clslib.LSTMEncoder(
             embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32), 
             **enc_kwargs
         )
    else:
        raise ValueError(f"I do not know this encoder type: {enc_type}")    

    model = clslib.SoftmaxRegressor(        
        encoder=encoder,
        **regressor_kwargs
    ).to(device)
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_kwargs['learning_rate'], weight_decay=opt_kwargs['weight_decay'])

    metrics = []
    for epoch in tqdm(range(opt_kwargs['num_epochs'])):
        train_metrics = training_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train acc: {train_metrics['accuracy']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"val acc: {val_metrics['accuracy']:.4f}"
            )
        
        # Saving the best model according to validation loss
        if epoch == 0 or val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            state = model.state_dict()
            if shared_table is not None and not model.encoder.embedding.weight.requires_grad:
                del state["encoder.embedding.weight"]
            torch.save(state, f"{save_to}.pt")

    time_training = time.time()
    print(f"Training done! Time taken: {time_training - time_dataloader:.3f} seconds")

    return metrics, best_epoch

    
def load(path, emb_model: clslib.PretrainedEmbeddings=None, device="cpu"):
    """
    Reconstruct the model architecture (using {path}.json) and load trained weights (using {path}.pt) and return it.
    """

    with open(f"{path}.json", "r", encoding="utf-8") as f:
        args = json.load(f)
    
    with open(f"{path}.pt", "rb") as f:
        state_dict = torch.load(f, map_location="cpu", weights_only=True)
    if "encoder.embedding.weight" not in state_dict:
        table_path = Path(path).parent / args["embedding_table"]
        table = torch.load(table_path, map_location="cpu", weights_only=True)
        state_dict["encoder.embedding.weight"] = table["vectors"]
        
    if isinstance(device, str):
        device = torch.device(device)
        
    if args['enc_type'] == 'embed-and-pool':
        encoder = clslib.EmbedAndPoolEncoder(
            embedding_matrix=state_dict['encoder.embedding.weight'],
            **args['enc_kwargs']
         )
    elif args['enc_type'] == 'lstm':
         encoder = clslib.LSTMEncoder(
             embedding_matrix=state_dict['encoder.embedding.weight'],
             **args['enc_kwargs']
         )
    else:
        raise ValueError(f"I do not know this encoder type: {args['enc_type']}")    

    model = clslib.SoftmaxRegressor(        
        encoder=encoder,
        **args['regressor_kwargs']
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

def cross_validation(
    outcome_space,
    data_splits, 
    tokenizer: clslib.Tokenizer, 
    emb_model: clslib.PretrainedEmbeddings, 
    enc_type, 
    enc_kwargs,
    regressor_kwargs, 
    opt_kwargs, 
    device='cpu',
    save_to="tmp",
    *,
    seed=None,
    shared_table=None,
):
    """
    Trains and evaluates k models using a k-fold data split for cross validation. 
    This function returns a dictionary with 
        - models: the paths to the k models trained (the model objects aren't returned, they are stored in disk)
        - predictions: the predictions by each of the k models
        - accuracies: the accuracy of each trained model on its corresponding validation partition
    
    data_splits: the training/validation folds
    encoder: an instance of Encoder
    model_option: for now this only supports 'logistic'
        which tells this function to use train_logistic_model (defined earlier in this notebook)
    save_to: folder where to store models
    smlp_args: dict with training arguments for smlp.train_neural_network_model
    """

    models, predictions, accuracies = OrderedDict(), OrderedDict(), OrderedDict()
    
    for fold in tqdm(data_splits.keys()):
        training_set = data_splits[fold]["training"]
        validation_set = data_splits[fold]["validation"]

        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True) 
        
        table_ref = os.path.relpath(shared_table, fold_folder) if shared_table is not None else None
        # Finish a fold before replacing any results from an earlier run.
        with TemporaryDirectory(dir=fold_folder) as temporary:
            staging = Path(temporary)
            metrics, best_epoch = train(
                training_set,
                validation_set,
                tokenizer,
                emb_model,
                num_classes=len(outcome_space),
                enc_type=enc_type,
                enc_kwargs=enc_kwargs,
                regressor_kwargs=regressor_kwargs,
                opt_kwargs=opt_kwargs,
                device=device,
                save_to=staging / "model",
                seed=seed,
                shared_table=table_ref,
            )
            best = metrics[best_epoch]["val_metrics"]
            models[fold] = str(fold_folder / "model")
            accuracies[fold] = best["accuracy"]
            predictions[fold] = np.array(list(itertools.chain.from_iterable(best["predictions"])))

            history = [
                {
                    "epoch": epoch,
                    "train_loss": record["train_metrics"]["loss"],
                    "train_accuracy": record["train_metrics"]["accuracy"],
                    "val_loss": record["val_metrics"]["loss"],
                    "val_accuracy": record["val_metrics"]["accuracy"],
                }
                for epoch, record in enumerate(metrics, start=1)
            ]
            summary = {
                "seed": seed,
                "fold": int(fold),
                "num_folds": len(data_splits),
                "opt_kwargs": opt_kwargs,
                "best_epoch": best_epoch + 1,
                "acc": best["accuracy"],
                "val_loss": best["loss"],
                "train_size": len(training_set),
                "val_size": len(validation_set),
                "history": history,
            }
            with (staging / "metric.json").open("w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2)

            if "snli_rows" in data_splits[fold]:
                row_ids = data_splits[fold]["snli_rows"]
                labels = itertools.chain.from_iterable(best["labels"])
                with gzip.open(staging / "predictions.csv.gz", "wt", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["snli_row", "label", "prediction"])
                    writer.writerows(zip(row_ids, labels, predictions[fold], strict=True))

            for artifact in staging.iterdir():
                artifact.replace(fold_folder / artifact.name)

    return {'models': models, 'predictions': predictions, 'accuracies': accuracies}    

def predict_logprobs(   
    model: clslib.SoftmaxRegressor,    
    tokenizer: clslib.Tokenizer,        
    texts,     
    batch_size: int,
    device='cpu'
):
    """Receives a training set and a validation set and returns a trained neural network model"""
        
    vectors = [tokenizer.ids(text) for text in texts]
    labels = [0 for _ in texts]

    pad_id = model.encoder.pad_idx

    dl = DataLoader(
        LabelledCorpus(vectors, labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_labelled_texts(batch, padding_value=pad_id),
    )

    if isinstance(device, str):
        device = torch.device(device)


    model.eval()

    preds_list = []
    labels_list = []
    logprobs_list = []
    with torch.no_grad():
        for x_batch, y_batch in dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logprobs = model(x_batch)
            logprobs_list.extend(logprobs.cpu().numpy())
    
    return np.array(logprobs_list)