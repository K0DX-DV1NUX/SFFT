import os
import subprocess

# Check if the logs directory exists, if not, create it
if not os.path.isdir("./logs"):
    os.mkdir("./logs")

# Define variables
model_name = "HADL"
root_path_name = "./dataset/"
data_path_name = "ETTh1.csv"
model_id_name = "ETTh1"
data_name = "ETTh1"
rank = 50

# Loop through the sequence lengths and prediction lengths
seq_lens = [512]
pred_lens = [96]

# Iterate over each combination of seq_len and pred_len
for seq_len in seq_lens:
    for pred_len in pred_lens:
        model_id = f"{model_id_name}_{seq_len}_{pred_len}"

        # Construct the command
        command = [
            "python", "-u", "run_longExp.py",
            "--is_training", "1",
            "--individual", "0",
            "--root_path", root_path_name,
            "--data_path", data_path_name,
            "--model_id", model_id,
            "--model", model_name,
            "--data", data_name,
            "--features", "S",
            "--train_type", "Linear",
            "--seq_len", str(seq_len),
            "--pred_len", str(pred_len),
            "--enc_in", "321",
            "--rank", str(rank),
            "--train_epochs", "50",
            "--bias", "1",
            "--enable_Haar", "1",
            "--enable_DCT", "1",
            "--enable_lowrank", "1",
            "--enable_iDCT", "0",
            "--patience", "10",
            "--des", "Exp",
            "--regularizer", "1",
            "--regularization_rate", "0.1",
            "--itr", "1",
            "--batch_size", "32",
            "--learning_rate", "0.01"
        ]
        
        # Run the command
        subprocess.run(command)
