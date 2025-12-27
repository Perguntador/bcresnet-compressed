import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import set_seed, initialize_weights, generate_test_report
from src.dataset import download_gsc_dataset, create_gsc_dataframes, SpeechCommandsDataset, PreprocessingDataLoader
from src.transforms import AudioPreprocessor
from src.bcresnet import BCResNets
from src.trainer import EarlyStopping, BCResNetScheduler, train_model

# --- Global Configurations ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directories ---
MODEL_DIR = "./models"
DATA_DIR = "./data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- Hyperparameters Grid ---
# Total combinations: 4 (TAU) x 5 (N) x 2 (HOP_LENGTH) = 40 experiments
N_SET = [24, 10, 4, 2, 1]
TAU_SET = [1.0, 0.5, 0.25, 0.125]
HOP_LENGTH_SET = [160, 320]

# --- Training Settings ---
VER = 2
BATCH_SIZE = 100
TOTAL_EPOCH = 200
WARMUP_EPOCH = 5
INIT_LR = 1e-1
LR_LOWER_LIMIT = 0
PATIENCE = 20

# --- Dataset Constants (from original paper) ---
# Indices: 0=Train, 1=Valid, 2=Test
SAMPLE_PER_CLS_V2 = [3077, 371, 408] 

VOCAB_CONFIGS = {
    1: ["marvin"], # Wake Word
    2: ["yes", "no"],
    4: ["up", "down", "left", "right"],
    10: ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
    24: [
        "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
        "backward", "forward", "follow", "learn",
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    ]
}

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    base_dir = download_gsc_dataset(destination_dir=DATA_DIR, version=VER, keep_archive=True)
    noise_dir = os.path.join(base_dir, "_background_noise_")
    
    # Create DataFrames only once
    df_train, df_valid, df_test, df_noise = create_gsc_dataframes(base_dir)
    print(f"Data Loaded - Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)} | Noise: {len(df_noise)}")

    limits_map = {
        'train': SAMPLE_PER_CLS_V2[0],
        'valid': SAMPLE_PER_CLS_V2[1],
        'test':  SAMPLE_PER_CLS_V2[2]
    }

    # 2. Iteration Loop (N -> TAU -> HOP_LENGTH)
    for N in N_SET:
        print(f"\n=== Starting experiments for N={N} classes ===")
        
        # Reset seed for dataset reproducibility
        set_seed(SEED)
        target_classes = VOCAB_CONFIGS[N]

        # Initialize Datasets (re-initialized per N because target_labels change)
        train_ds = SpeechCommandsDataset(
            df_train, target_classes, base_dir, mode='train', 
            cache_ram=True, limit_unknown=limits_map['train'], 
            limit_silence=limits_map['train'], random_state=SEED
        )
        valid_ds = SpeechCommandsDataset(
            df_valid, target_classes, base_dir, mode='valid', 
            cache_ram=True, limit_unknown=limits_map['valid'], 
            limit_silence=limits_map['valid'], random_state=SEED
        )
        test_ds = SpeechCommandsDataset(
            df_test, target_classes, base_dir, mode='test', 
            cache_ram=True, limit_unknown=limits_map['test'], 
            limit_silence=limits_map['test'], random_state=SEED
        )

        # Loaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        for TAU in TAU_SET:
            # Determine SpecAugment parameters based on model size (TAU)
            use_specaugment = TAU >= 1.5
            if TAU < 1.5: freq_mask_para = 0
            elif TAU < 2.0: freq_mask_para = 1
            elif TAU < 3.0: freq_mask_para = 3
            elif TAU < 6.0: freq_mask_para = 5
            else: freq_mask_para = 7

            for HOP_LENGTH in HOP_LENGTH_SET:
                # Reset seed for training stability across configurations
                set_seed(SEED)
                
                # Ensure dataset internal RNG is reset if needed
                train_ds.set_random_state(SEED)
                valid_ds.set_random_state(SEED)
                test_ds.set_random_state(SEED)

                config_name = f"v{VER}_N{N}_tau{TAU}_hl{HOP_LENGTH}_seed{SEED}"
                print(f"\n--- Running Config: {config_name} ---")

                # Preprocessors
                preprocess_train = AudioPreprocessor(
                    device=DEVICE, noise_dir=noise_dir, specaug=use_specaugment,
                    frequency_masking_para=freq_mask_para, hop_length=HOP_LENGTH
                )
                preprocess_eval = AudioPreprocessor(
                    device=DEVICE, noise_dir=noise_dir, specaug=False, hop_length=HOP_LENGTH
                )

                # Wrapped Loaders
                train_iter = PreprocessingDataLoader(train_loader, preprocess_train, is_train=True)
                valid_iter = PreprocessingDataLoader(valid_loader, preprocess_eval, is_train=False)
                test_iter = PreprocessingDataLoader(test_loader, preprocess_eval, is_train=False)

                # Model Initialization
                model = BCResNets(int(TAU * 8), num_classes=(len(target_classes) + 2)).to(DEVICE)
                model.apply(initialize_weights)
                
                param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"BC-ResNet-{TAU} initialized. Parameters: {param_count:,}")

                # Training Setup
                optimizer = torch.optim.SGD(model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9)
                scheduler = BCResNetScheduler(optimizer, len(train_loader), TOTAL_EPOCH, WARMUP_EPOCH, INIT_LR, LR_LOWER_LIMIT)
                criterion = F.cross_entropy
                
                early_stopping_path = os.path.join(MODEL_DIR, f"temp_checkpoint_{config_name}.pt")
                early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=early_stopping_path)

                # Train Loop
                start_time = time.time()
                model, train_loss, valid_loss, train_acc, valid_acc = train_model(
                    model, early_stopping, TOTAL_EPOCH, optimizer, 
                    train_iter, valid_iter, criterion, scheduler, DEVICE
                )
                duration = time.time() - start_time
                print(f"Training finished in {int(duration // 3600):02d}:{int((duration % 3600) // 60):02d}:{int(duration % 60):02d}")

                # Save Final Model and History
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"bcresnet_{config_name}.pt"))
                torch.save({
                    'train_loss': train_loss, 'valid_loss': valid_loss,
                    'train_acc': train_acc, 'valid_acc': valid_acc,
                    'duration': duration, 'params': param_count
                }, os.path.join(MODEL_DIR, f"history_{config_name}.pt"))
                
                print(f"Model and history saved to {MODEL_DIR}")

                # Final Test Report
                generate_test_report(model, test_iter, DEVICE)

                # Cleanup temp file
                if os.path.exists(early_stopping_path):
                    os.remove(early_stopping_path)

if __name__ == "__main__":
    main()