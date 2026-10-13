import torchaudio, pandas, os, torch
from torch.utils.data import Dataset


class MSP2_Dataset(Dataset):
    def __init__(self, split="Train",
                 path="/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Labels/labels_consensus.csv",
                 root_path = "/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Audios",
                 duration = 6):
        super().__init__()
        if split not in ['Test1', 'Train', 'Development', 'Test2']:
            raise ValueError("Split must be in 'Test1' 'Train' 'Development' 'Test2', YOU STUPID")

        df = pandas.read_csv(path, usecols=["FileName", "EmoClass", "Split_Set"])
        self.df = df[df["Split_Set"]==split]
        self.df = self.df[self.df["EmoClass"].isin(['A', 'C', 'D', 'F', 'H', 'N', 'U', 'S'])].reset_index(drop=True)
        self.labels = sorted(self.df["EmoClass"].unique())
        self.labels_dict = {k: v for k,v in zip(self.labels, range(len(self.labels)))}
        self.audio_path = root_path
        self.duration = int(16000*duration)

    def __len__(self):
        return len(self.df)
        # return 100
        
    
    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(dim=0, keepdim=True)                        
        wav = wav.repeat(1, self.duration // wav.shape[-1] + 1)     
        return wav[:, :self.duration]       
    
    def __getitem__(self, index):
        audio = self.load_audio(path=os.path.join(self.audio_path, self.df.iloc[index]["FileName"]))
        label = self.labels_dict[self.df.iloc[index]["EmoClass"]]
        return audio, label
    
    
    
class MSP2_Dataset_Batch_Balance(Dataset):
    def __init__(self, split="Train",
             path="/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Labels/labels_consensus.csv",
             root_path="/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Audios",
             duration=6):
        super().__init__()
        if split not in ['Test1', 'Train', 'Development', 'Test2']:
            raise ValueError("Split must be in 'Test1' 'Train' 'Development' 'Test2', YOU STUPID")

        df = pandas.read_csv(path, usecols=["FileName", "EmoClass", "Split_Set"])
        self.df = df[df["Split_Set"] == split]
        self.df = self.df[self.df["EmoClass"].isin(['A', 'C', 'D', 'F', 'H', 'N', 'U', 'S'])].reset_index(drop=True)
        self.labels = sorted(self.df["EmoClass"].unique())
        self.labels_dict = {k: v for k, v in zip(self.labels, range(len(self.labels)))}
        self.label_list = self.df["EmoClass"].map(self.labels_dict).tolist()  # ← ADD THIS
        self.audio_path = root_path
        self.duration = int(16000 * duration)

    def __len__(self):
        return len(self.df)
        # return 100
        
    
    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(dim=0, keepdim=True)                        
        wav = wav.repeat(1, self.duration // wav.shape[-1] + 1)     
        return wav[:, :self.duration]       
    
    def __getitem__(self, index):
        audio = self.load_audio(path=os.path.join(self.audio_path, self.df.iloc[index]["FileName"]))
        label = self.labels_dict[self.df.iloc[index]["EmoClass"]]
        return audio, label
    
class MSP2_DatasetKL(Dataset):
    CLASSES = ['A', 'C', 'D', 'F', 'H', 'N', 'S', 'U']   # 8 classes, fixed order

    def __init__(self,
                 split       = "Train",
                 label_path  = "/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Labels/labels_consensus.csv",
                 soft_path   = "/home/hiddenrock/SpeechEmotionRecog/IS-25/MSP_SER/soft_labels.csv",
                 root_path   = "/home/hiddenrock/SERVER_DATASETS/DATASET_MSP2/Audios",
                 duration    = 6):
        super().__init__()
        if split not in ['Test1', 'Train', 'Development', 'Test2']:
            raise ValueError("Split must be in 'Test1' 'Train' 'Development' 'Test2'")

        # ── consensus CSV: used for split assignment only ────────────────────
        df = pandas.read_csv(label_path, usecols=["FileName", "EmoClass", "Split_Set"])
        df = df[df["Split_Set"] == split]
        df = df[df["EmoClass"].isin(self.CLASSES)].reset_index(drop=True)

        # ── soft labels: (N_files × 8) probability matrix ───────────────────
        soft_df = pandas.read_csv(soft_path).set_index("FileName")

        # keep only files that have soft labels
        df = df[df["FileName"].isin(soft_df.index)].reset_index(drop=True)

        self.df         = df
        self.soft_df    = soft_df[self.CLASSES]          # enforce column order
        self.labels     = self.CLASSES
        self.labels_dict = {k: v for v, k in enumerate(self.CLASSES)}
        self.audio_path = root_path
        self.duration   = int(16000 * duration)

    def __len__(self):
        return len(self.df)

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(dim=0, keepdim=True)
        wav = wav.repeat(1, self.duration // wav.shape[-1] + 1)
        return wav[:, :self.duration]

    def __getitem__(self, index):
        row       = self.df.iloc[index]
        audio     = self.load_audio(os.path.join(self.audio_path, row["FileName"]))
        soft_lbl  = torch.tensor(
            self.soft_df.loc[row["FileName"]].values.astype("float32")
        )                                               # shape: (8,), sums to 1
        return audio, soft_lbl