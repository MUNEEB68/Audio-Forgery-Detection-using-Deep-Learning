# Path to your label file
label_file = r"D:\data\HAD\HAD_train\HAD_train_label.txt"

# Path to save the new labels file
target_file = r"target_labels.txt"

with open(label_file, 'r') as f_in, open(target_file, 'w') as f_out:
    for line in f_in:
        line = line.strip()
        if line == "":
            continue
        parts = line.split()
        label = parts[-1]  # last element is the label
        name_of_audio_file = parts[0] # first element is the audio file name
        f_out.write(f"{name_of_audio_file}.wav {label}\n")

        

print(f"Target labels saved to: {target_file}")