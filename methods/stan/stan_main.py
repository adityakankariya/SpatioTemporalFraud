import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from math import floor, ceil
from methods.stan.stan import stan_model
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, average_precision_score
from torch.nn.utils import prune

from utils import count_nonzero_parameters, fine_tune, eval_model


def to_pred(logits: torch.Tensor) -> list:
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()

def att_train(
    x_train,
    y_train,
    x_test,
    y_test,
    num_classes: int = 2,
    epochs: int = 18,
    batch_size: int = 256,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu"
):
    model = stan_model(
        time_windows_dim=x_train.shape[1],
        spatio_windows_dim=x_train.shape[2],
        feat_dim=x_train.shape[3],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
    )
    model.to(device)

    nume_feats = x_train
    labels = y_train

    nume_feats.requires_grad = False
    labels.requires_grad = False

    nume_feats.to(device)
    labels = labels.to(device)

    # anti label imbalance
    unique_labels, counts = torch.unique(labels, return_counts=True)
    weights = (1 / counts)*len(labels)/len(unique_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weights)

    batch_num = ceil(len(labels) / batch_size)
    for epoch in range(epochs):

        loss = 0.
        pred = []

        for batch in (range(batch_num)):
            optimizer.zero_grad()

            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(labels))))

            output = model(nume_feats[batch_mask])

            batch_loss = loss_func(output, labels[batch_mask])
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            pred.extend(to_pred(output))

        true = labels.cpu().numpy()
        pred = np.array(pred)
        print(
            f"Epoch: {epoch}, loss: {(loss / batch_num):.4f}, auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")

    feats_test = x_test
    labels_test = y_test

    batch_num_test = ceil(len(labels_test) / batch_size)
    with torch.no_grad():
        pred = []
        for batch in range(batch_num_test):
            optimizer.zero_grad()
            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(labels_test))))
            output = model(feats_test[batch_mask])
            pred.extend(to_pred(output))

        true = labels_test.cpu().numpy()
        pred = np.array(pred)
        print(
            f"test set | auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")
        print(confusion_matrix(true, pred))

        return model

def stan_train(
    train_feature_dir,
    train_label_dir,
    test_feature_dir,
    test_label_dir,
    save_path: str,
    mode: str = "3d",
    epochs: int = 18,
    batch_size: int = 256,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu"
):
    train_feature = torch.from_numpy(np.load(train_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    train_label = torch.from_numpy(np.load(train_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)
    test_feature = torch.from_numpy(np.load(test_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    test_label = torch.from_numpy(np.load(test_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)

    # y_pred = np.zeros(shape=test_label.shape)
    if mode == "3d":
        model = att_train(
            train_feature,
            train_label,
            test_feature,
            test_label,
            epochs=epochs,
            batch_size=batch_size,
            attention_hidden_dim=attention_hidden_dim,
            lr=lr,
            device=device
        )

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

def stan_test(
    test_feature_dir,
    test_label_dir,
    path: str,
    num_classes: int = 2,
    batch_size: int = 256,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu",
):
    x_test = torch.from_numpy(np.load(test_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_test = torch.from_numpy(np.load(test_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)

    model = stan_model(
        time_windows_dim=x_test.shape[1],
        spatio_windows_dim=x_test.shape[2],
        feat_dim=x_test.shape[3],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
    )
    if device == "cpu":
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(path))
    model.to(device)

    feats_test = x_test
    labels_test = y_test

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_num = ceil(len(labels_test) / batch_size)
    with torch.no_grad():
        pred = []
        for batch in range(batch_num):
            optimizer.zero_grad()
            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(labels_test))))
            output = model(feats_test[batch_mask])
            pred.extend(to_pred(output))

        true = labels_test.cpu().numpy()
        pred = np.array(pred)
        print(
            f"test set | auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")
        # print(confusion_matrix(true, pred))

def stan_prune(
        train_feature_dir,
        train_label_dir,
        test_feature_dir,
        test_label_dir,
        load_path: str,
        batch_size=256,
        attention_hidden_dim=150,
        lr=3e-3,
        num_classes=2,
        device='cpu',
        fine_tune_epochs=4,
        prune_iter=3,
        prune_perct=0.1
    ):

    x_train = torch.from_numpy(np.load(train_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_train = torch.from_numpy(np.load(train_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)
    x_test = torch.from_numpy(np.load(test_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_test = torch.from_numpy(np.load(test_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)
    
    model = stan_model(
        time_windows_dim=x_test.shape[1],
        spatio_windows_dim=x_test.shape[2],
        feat_dim=x_test.shape[3],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
    )
    if device == "cpu":
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(load_path))
    model.to(device)

    print(f"Number of parameters in original model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Iterative pruning
    tmp = prune_iter
    while prune_iter > 0:
        prune_iter -= 1

        # Prune the Conv3d layer
        prune.l1_unstructured(model.conv, name='weight', amount=prune_perct)

        # Prune each Linear layer within 'linears'
        for name, module in model.named_children():
            if name == 'linears':
                for name_seq, module_seq in module.named_children():
                    if isinstance(module_seq, torch.nn.Linear):
                        prune.l1_unstructured(module_seq, name='weight', amount=prune_perct)


        # Retrain to regain lost accuracy
        fine_tune(model, x_train, y_train, batch_size, lr, device, fine_tune_epochs)
        eval_model(model, x_test, y_test, batch_size, lr)
        print("*" * 3 + f" Prune iteration {tmp - prune_iter} complete " + "*" * 3)
    
    # Make pruning permanent
    for name, module in model.named_modules():
        for hook in list(module._forward_pre_hooks.values()):
            if isinstance(hook, torch.nn.utils.prune.BasePruningMethod):
                prune.remove(module, 'weight')

    print(f"Number of parameters in pruned model: {count_nonzero_parameters(model)} parameters")

    # save model
    torch.save(model.state_dict(), load_path.replace('.pt', '_pruned.pt'))

    