import os, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse, torch, pandas
import torch.nn as nn
import torch.nn.functional as F
import torchaudio, yaml

from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast

from MSP_SER.dataset import MSP2_Dataset
from MSP_SER.criterion import WeightedFocalLoss
from MSP_SER.utilities import GetYAMLConfigs, Logs
from MSP_SER.architectures import WavLMLayerWeighted, ProjectionNetwork


###############################################################################
###############################################################################

P = argparse.ArgumentParser()
P.add_argument("--config", type=str, required=True)
A = P.parse_args()

configs = GetYAMLConfigs(path=A.config)
logger = Logs(exp_name=configs["experiment_name"])
DEVICE = torch.device("cuda:{}".format(configs["cuda_device"]))

torch.cuda.empty_cache()
torch.manual_seed(configs["seed"])

###############################################################################
###############################################################################

train_dataloader = DataLoader(dataset=MSP2_Dataset(), batch_size=configs["BATCH_SIZE"], shuffle=True)
dev_dataloader = DataLoader(dataset=MSP2_Dataset(split="Development"), batch_size=configs["BATCH_SIZE"], shuffle=True)

feature_extractor = WavLMLayerWeighted(k=6).to(DEVICE)
projection_network = ProjectionNetwork(dims=feature_extractor.final_dim).to(DEVICE)
criterion = WeightedFocalLoss(num_classes=len(train_dataloader.dataset.labels)).to(DEVICE)

# optimizer = torch.optim.AdamW(
#     list(feature_extractor.parameters())+list(projection_network.parameters()),
#     lr=configs["learning_rate"],
#     weight_decay=0.01
# )
optimizer = torch.optim.AdamW(
    [
        {"params": feature_extractor.parameters(), "lr": configs["learning_rate"] * 0.1},
        {"params": projection_network.parameters(), "lr": configs["learning_rate"]},
    ],
    weight_decay=0.01,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=9)
scaler = torch.amp.GradScaler(f"cuda:{configs["cuda_device"]}")
accum_steps = configs["accumulation_steps"]


for e in range(configs["epochs"]):
    feature_extractor.train()
    projection_network.train()
    criterion.train()
    
    epoch_loss = 0
    epoch_total = 0
    epoch_correct = 0
    
    for mb_idx, (mb_data, mb_label) in enumerate(train_dataloader):
        mb_data, mb_label = mb_data.squeeze(1).to(DEVICE), mb_label.to(DEVICE)

        with autocast(device_type=f"cuda:{configs["cuda_device"]}"):
            embeddings = feature_extractor(mb_data)
            predictions = projection_network(embeddings)
            loss = criterion(predictions, mb_label) / accum_steps

        scaler.scale(loss).backward()

        if (mb_idx + 1) % accum_steps == 0 or (mb_idx + 1) == len(train_dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(feature_extractor.parameters()) + list(projection_network.parameters()), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accum_steps
        preds = predictions.argmax(dim=-1)
        epoch_correct += (preds == mb_label).sum().item()
        epoch_total += mb_label.size(0)

        if mb_idx % configs["minibatch_log_after"] == 0:
            logger.write("Epoch: [{}], MiniBatch: [{}], Loss: [{}], Accuracy: [{:.2f}%]".format(
                e, mb_idx, loss.item() * accum_steps, 100 * epoch_correct / epoch_total
            ))

    scheduler.step()
    logger.write("\n=== Epoch: [{}] | Avg Loss: [{:.4f}] | Accuracy: [{:.2f}%] ===".format(
        e, epoch_loss / len(train_dataloader), 100 * epoch_correct / epoch_total
    ))
    
    
    feature_extractor.eval()
    projection_network.eval()

    dev_loss, dev_correct, dev_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for mb_data, mb_label in dev_dataloader:
            mb_data, mb_label = mb_data.squeeze(1).to(DEVICE), mb_label.to(DEVICE)
            with autocast(device_type="cuda"):
                embeddings  = feature_extractor(mb_data)
                predictions = projection_network(embeddings)
                loss        = criterion(predictions, mb_label)
            dev_loss    += loss.item()
            preds        = predictions.argmax(dim=-1)
            dev_correct += (preds == mb_label).sum().item()
            dev_total   += mb_label.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(mb_label.cpu().tolist())

    logger.write("=== DEV Epoch: [{}] | Avg Loss: [{:.4f}] | Accuracy: [{:.2f}%] | "
                "Precision (Micro): [{:.4f}] | Precision (Macro): [{:.4f}] | "
                "Recall (Micro): [{:.4f}] | Recall (Macro): [{:.4f}] | "
                "Recall (Weighted): [{:.4f}] | "
                "F1-Micro: [{:.4f}] | F1-Macro: [{:.4f}] ===".format(
        e,
        dev_loss / len(dev_dataloader),
        100 * dev_correct / dev_total,
        precision_score(all_labels, all_preds, average="micro", zero_division=0),   # Precision (Micro)
        precision_score(all_labels, all_preds, average="macro", zero_division=0),   # Precision (Macro)
        recall_score(all_labels, all_preds, average="micro"),                        # Recall (Micro / Unweighted)
        recall_score(all_labels, all_preds, average="macro"),                        # Recall (Macro / Unweighted avg)
        recall_score(all_labels, all_preds, average="weighted"),                     # Recall (Weighted)
        f1_score(all_labels, all_preds, average="micro"),                            # F1-Micro
        f1_score(all_labels, all_preds, average="macro"),                            # F1-Macro
    ))

    torch.save(obj={
            "feature_extractor": feature_extractor.state_dict(),
            "projection_network": projection_network.state_dict(),
        },
               f=os.path.join(logger.chkpt_folder, f"epoch_{e}.pt"))

    if (e + 1) % configs["compute_test_scores_after"] == 0:
        for split in ["Test1", "Test2"]:
            test_dataloader = DataLoader(dataset=MSP2_Dataset(split=split), batch_size=configs["BATCH_SIZE"], shuffle=False)
            t_loss, t_correct, t_total = 0.0, 0, 0
            t_preds, t_labels = [], []

            with torch.no_grad():
                for mb_data, mb_label in test_dataloader:
                    mb_data, mb_label = mb_data.squeeze(1).to(DEVICE), mb_label.to(DEVICE)
                    with autocast(device_type="cuda"):
                        embeddings  = feature_extractor(mb_data)
                        predictions = projection_network(embeddings)
                        loss        = criterion(predictions, mb_label)
                    t_loss    += loss.item()
                    preds      = predictions.argmax(dim=-1)
                    t_correct += (preds == mb_label).sum().item()
                    t_total   += mb_label.size(0)
                    t_preds.extend(preds.cpu().tolist())
                    t_labels.extend(mb_label.cpu().tolist())
                    
                logger.write("=== {} Epoch: [{}] | Avg Loss: [{:.4f}] | Accuracy: [{:.2f}%] | "
                            "Precision (Micro): [{:.4f}] | Precision (Macro): [{:.4f}] | "
                            "Recall (Micro): [{:.4f}] | Recall (Macro): [{:.4f}] | "
                            "Recall (Weighted): [{:.4f}] | "
                            "F1-Micro: [{:.4f}] | F1-Macro: [{:.4f}] ===".format(
                    split,
                    e,
                    t_loss / len(test_dataloader),
                    100 * t_correct / t_total,
                    precision_score(t_labels, t_preds, average="micro", zero_division=0),   # Precision (Micro)
                    precision_score(t_labels, t_preds, average="macro", zero_division=0),   # Precision (Macro)
                    recall_score(t_labels, t_preds, average="micro"),                        # Recall (Micro)
                    recall_score(t_labels, t_preds, average="macro"),                        # Recall (Macro / Unweighted)
                    recall_score(t_labels, t_preds, average="weighted"),                     # Recall (Weighted)
                    f1_score(t_labels, t_preds, average="micro"),                            # F1-Micro
                    f1_score(t_labels, t_preds, average="macro"),                            # F1-Macro
                ))
        logger.write("\n")


with open("/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Partitions.txt", "r") as file:
    test3_lines = [os.path.join("/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Audios", line.strip().replace("Test3; ", "")) for line in file.readlines() if "Test3" in line]

inv_labels = {v: k for k, v in train_dataloader.dataset.labels_dict.items()}
duration   = int(16000 * 6)

def load_audio(path, full_audio):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0, keepdim=True)
    wav = wav.repeat(1, duration // wav.shape[-1] + 1)
    if full_audio:
        return wav.mean(dim=0, keepdim=True)
    else:
        return wav[:, :duration]

feature_extractor.eval()
projection_network.eval()

filenames, pred_classes = [], []
batch_size = configs["BATCH_SIZE"]

with torch.no_grad():
    for i in range(0, len(test3_lines), batch_size):
        batch_paths = test3_lines[i:i + batch_size]
        mb_data     = torch.stack([load_audio(p, full_audio=True) for p in batch_paths]).squeeze(1).to(DEVICE)

        with autocast(device_type="cuda"):
            preds = projection_network(feature_extractor(mb_data)).argmax(dim=-1)

        filenames.extend([os.path.basename(p) for p in batch_paths])
        pred_classes.extend([inv_labels[p.item()] for p in preds])

pandas.DataFrame({"FileName": filenames, "EmoClass": pred_classes}).to_csv(os.path.join(logger.chkpt_folder, "test3_predictions.csv"), index=False)
logger.write("Test3 predictions saved → test3_predictions.csv ({} samples)".format(len(filenames)))
