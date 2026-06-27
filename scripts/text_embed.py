import os.path as osp
import json
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

model_name = "/root/autodl-tmp/Text-MedSAM/ckpts/biomedbert"
json_path = "/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/CVPR25_TextSegFMData_with_class.json"
embedding_path = "/root/autodl-tmp/Text-MedSAM/CVPR-BiomedSegFM/"
output_pt_path = osp.join(embedding_path, "text_embed.pt")

tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/Text-MedSAM/ckpts/biomedclip", local_files_only=True)
device = torch.device("cuda")
model = BertModel.from_pretrained(model_name)
model.load_state_dict(torch.load("/root/autodl-tmp/Text-MedSAM/ckpts/bert.pt"), strict=False)
model.to(device).eval()

with open(json_path, 'r') as f:
    data = json.load(f)

dataset2id = {}
class2id = {}
temp_embed = {}
max_classes = 0
max_texts = 0
embed_dim = None
seq_len = None

with torch.no_grad():
    for d_id, (dataset_name, dataset_info) in enumerate(tqdm(data.items(), desc="Encoding Texts")):
        dataset2id[dataset_name] = d_id
        class2id[d_id] = {}
        temp_embed[d_id] = {}
        is_instance = bool(dataset_info["instance_label"])
        class_keys = [k for k in dataset_info.keys() if k != "instance_label"]
        max_classes =  max(max_classes, len(class_keys))
        c_id = 0
        for label_id, texts in dataset_info.items():
            if label_id == "instance_label":
                continue
            if is_instance:
                label_id = "1"
            print(texts)
            token_ids = tokenizer(
                texts,
                truncation=True,
                max_length=256,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            ).to(model.device)

            # last_hidden_state
            text_features = model(
                input_ids=token_ids["input_ids"],
                attention_mask=token_ids['attention_mask']
            )["last_hidden_state"]
            text_features = text_features.cpu() 
            
            max_texts = max(max_texts, text_features.shape[0])
            if embed_dim is None:
                seq_len = text_features.shape[1]
                embed_dim = text_features.shape[2]
                
            class2id[d_id][label_id] = c_id
            temp_embed[d_id][c_id] = text_features
            c_id += 1

num_datasets = len(data)
print("\n>>> Finished!")
print(f"  -  (Num_datasets): {num_datasets}")
print(f"  -  (Max_classes): {max_classes}")
print(f"  -  (Max_texts): {max_texts}")
print(f"  -  (Seq_Len): {seq_len}")
print(f"  -  (Embed_dim): {embed_dim}")

unified_embeds = torch.zeros((num_datasets, max_classes, max_texts, seq_len, embed_dim), dtype=torch.float32)
valid_text_counts = torch.zeros((num_datasets, max_classes), dtype=torch.long)

for d_id in range(num_datasets):
    for c_id, embeds in temp_embed[d_id].items():
        num_t = embeds.shape[0]
        unified_embeds[d_id, c_id, :num_t, :, :] = embeds
        valid_text_counts[d_id, c_id] = num_t
        
torch.save({
    'embeddings': unified_embeds,          # shape: (N_ds, Max_cls, Max_txt, 256, 768)
    'valid_text_counts': valid_text_counts,# shape: (N_ds, Max_cls)
    'dataset2id': dataset2id,              # dict
    'class2id': class2id                   # nested dict
}, output_pt_path)

print(f"text_embed_mapper saved to {output_pt_path}!")