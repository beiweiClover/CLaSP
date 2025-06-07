import numpy as np
import pandas as pd
import pickle as pkl
import random
import os
from tqdm import tqdm, trange
from threading import Thread, Lock
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, \
    jaccard_score, balanced_accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from unicore.modules import init_bert_params
from unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset, PrependTokenDataset,
    AppendTokenDataset, FromNumpyDataset, RightPadDataset, RightPadDataset2D,
    RawArrayDataset, RawLabelDataset,
)
from unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord,
)
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer
import wandb
import csv
from torch.amp import autocast, GradScaler

def set_random_seed(random_seed=1024):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    th.manual_seed(random_seed)
    th.cuda.manual_seed(random_seed)
    th.cuda.manual_seed_all(random_seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.enabled = False

def calculate_molecule_3D_structure(csv_data_path, data_dir):
    os.makedirs(f'{data_dir}/intermediate', exist_ok=True)
    
    def get_smiles_list_():
        data_df = pd.read_csv(csv_data_path)
        smiles_list = data_df["SMILES"].tolist()
        smiles_list = list(set(smiles_list))
        print(f"Total unique SMILES: {len(smiles_list)}")
        return smiles_list

    def calculate_molecule_3D_structure_(smiles_list):
        n = len(smiles_list)
        global p
        index = 0
        while True:
            mutex.acquire()
            if p >= n:
                mutex.release()
                break
            index = p
            p += 1
            mutex.release()

            smiles = smiles_list[index]
            print(f"Processing SMILES {index}/{n}: {smiles}")

            molecule = Chem.MolFromSmiles(smiles)
            molecule = AllChem.AddHs(molecule)
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42, useRandomCoords=True, maxAttempts=1000)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                with open(f'{data_dir}/intermediate/invalid_smiles.txt', 'a') as f:
                    f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                with open(f'{data_dir}/intermediate/invalid_smiles.txt', 'a') as f:
                    f.write('MMFFOptimizeMolecule error' + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            coordinates = molecule.GetConformer().GetPositions()

            assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
            coordinate_list.append(coordinates.astype(np.float32))

            global smiles_to_conformation_dict
            mutex.acquire()
            smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
            mutex.release()

    mutex = Lock()
    invalid_smiles_path = f'{data_dir}/intermediate/invalid_smiles.txt'
    if os.path.exists(invalid_smiles_path):
        os.remove(invalid_smiles_path)
        
    smiles_list = get_smiles_list_()
    global smiles_to_conformation_dict
    smiles_to_conformation_dict = {}
    global p
    p = 0
    thread_count = 16
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_molecule_3D_structure_, args=(smiles_list,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    pkl.dump(smiles_to_conformation_dict,
             open(f'{data_dir}/intermediate/smiles_to_conformation_dict.pkl', 'wb'))
    print('Valid smiles count:', len(smiles_to_conformation_dict))

def construct_data_list(data_dir):
    data_df = pd.read_csv(f"{data_dir}/{os.path.basename(data_dir)}.csv")
    smiles_to_conformation_dict = pkl.load(
        open(f'{data_dir}/intermediate/smiles_to_conformation_dict.pkl', 'rb'))
    
    label_columns = {
        'F50': -1,                    
        'PPB': -1,                    
        'CYP3A4_substrate': -1,       
        'CYP3A4_inhibitor': -1,       
        'CYP2D6_substrate': -1,       
        'CYP2C9_inhibitor': -1,       
        'CLp_c': -1,                  
        'FDAMDD_c': -1,               
        'DILI': -1,                   
        'Micronucleus': -1,          
        'Reproductive_toxicity': -1,  
        'Ames': -1,                   
        'BSEP_inhibitor': -1,         
        'Pgp_inhibitor': -1,          
        'Neurotoxicity': -1,          
        'VDss': float('nan'),         
    }
    
    stats = {label: {'valid': 0, 'missing': 0} for label in label_columns}
    
    data_list = []
    for index, row in data_df.iterrows():
        smiles = row["SMILES"]
        if smiles in smiles_to_conformation_dict:
            data_item = {
                "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                "smiles": smiles,
                "dataset_type": row["dataset_type"]
            }

            for label, default_value in label_columns.items():
                value = row.get(label)
                if pd.notna(value):
                    data_item[label] = float(value)
                    data_item[f"{label}_mask"] = 1.0
                    stats[label]['valid'] += 1
                else:
                    data_item[label] = default_value
                    data_item[f"{label}_mask"] = 0.0
                    stats[label]['missing'] += 1
            
            data_list.append(data_item)
    
    print("\nData Statistics:")
    print("-" * 50)
    for label, stat in stats.items():
        total = stat['valid'] + stat['missing']
        valid_percent = (stat['valid'] / total * 100) if total > 0 else 0
        print(f"{label}:")
        print(f"  Valid: {stat['valid']} ({valid_percent:.2f}%)")
        print(f"  Missing: {stat['missing']}")
    print("-" * 50)
    
    if len(data_list) == 0:
        raise ValueError("No valid data found after processing")
    
    pkl.dump(data_list, open(f'{data_dir}/intermediate/data_list.pkl', 'wb'))
    print(f"Processed {len(data_list)} valid entries")
    
    return data_list

def validate_data_splits(data_list_train, data_list_validate, data_list_test):
    total_size = len(data_list_train) + len(data_list_validate) + len(data_list_test)
    if total_size == 0:
        raise ValueError("Total dataset size is 0")
        
    min_size = 10
    for name, dataset in [
        ("Training", data_list_train),
        ("Validation", data_list_validate),
        ("Test", data_list_test)
    ]:
        if len(dataset) < min_size:
            raise ValueError(f"{name} set too small: {len(dataset)} samples "
                           f"(minimum required: {min_size})")
    
    train_ratio = len(data_list_train) / total_size
    val_ratio = len(data_list_validate) / total_size
    test_ratio = len(data_list_test) / total_size
    
    print("\nDataset Split Statistics:")
    print("-" * 50)
    print(f"Total samples: {total_size}")
    print(f"Train set: {len(data_list_train)} ({train_ratio:.2%})")
    print(f"Validation set: {len(data_list_validate)} ({val_ratio:.2%})")
    print(f"Test set: {len(data_list_test)} ({test_ratio:.2%})")
    
    def check_label_distribution(dataset, name):
        label_stats = {}
        for item in dataset:
            for label in item.keys():
                if label.endswith('_mask'):
                    base_label = label[:-5]
                    if base_label not in label_stats:
                        label_stats[base_label] = {'valid': 0, 'total': 0}
                    label_stats[base_label]['total'] += 1
                    if item[label] > 0:
                        label_stats[base_label]['valid'] += 1
        
        print(f"\n{name} Label Distribution:")
        for label, stats in label_stats.items():
            valid_ratio = stats['valid'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{label}: {stats['valid']}/{stats['total']} ({valid_ratio:.2%} valid)")
    
    for name, dataset in [
        ("Training", data_list_train),
        ("Validation", data_list_validate),
        ("Test", data_list_test)
    ]:
        check_label_distribution(dataset, name)
    
    return True

def convert_data_list_to_data_loader(data_dir):
    def convert_data_list_to_dataset_(data_list):
        dictionary = Dictionary.load('../data/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        
        label_distributions = {
            'F50': set(),
            'PPB': set(), 
            'CYP3A4_substrate': set(),
            'CYP3A4_inhibitor': set(),
            'CYP2D6_substrate': set(),
            'CYP2C9_inhibitor': set(),
            'CLp_c': set(),
            'FDAMDD_c': set(),
            'DILI': set(),
            'Micronucleus': set(),
            'Reproductive_toxicity': set(),
            'Ames': set(),
            'BSEP_inhibitor': set(),
            'Pgp_inhibitor': set(),
            'Neurotoxicity': set(),
            'VDss': set()
        }
        smiles_dataset = KeyDataset(data_list, "smiles")
        label_datasets = {}
        mask_datasets = {}
        
        for label in label_distributions.keys():
            label_data = [item[label] for item in data_list]
            mask_data = [item[f"{label}_mask"] for item in data_list]
            
            label_datasets[label] = RawLabelDataset(label_data)
            mask_datasets[f"{label}_mask"] = RawLabelDataset(mask_data)
            
        dataset = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
        dataset = AtomTypeDataset(data_list, dataset)
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, False)
        dataset = CroppingDataset(dataset, 1, "atoms", "coordinates", 256)
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=512)
        coord_dataset = KeyDataset(dataset, "coordinates")
        src_dataset = AppendTokenDataset(PrependTokenDataset(token_dataset, dictionary.bos()), dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = AppendTokenDataset(PrependTokenDataset(coord_dataset, 0.0), 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        return NestedDictionaryDataset({
            "input": {
                "src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad(), ),
                "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0, ),
                "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0, ),
                "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0, ),
                "smiles": RawArrayDataset(smiles_dataset),
            },
            "target": {
                **label_datasets,
                **mask_datasets,
            }
        })

    batch_size = 6
    data_list = pkl.load(open(f'{data_dir}/intermediate/data_list.pkl', 'rb'))
    
    data_list_train = [data_item for data_item in data_list if data_item["dataset_type"] == "train"]
    data_list_validate = [data_item for data_item in data_list if data_item["dataset_type"] == "val"]
    data_list_test = [data_item for data_item in data_list if data_item["dataset_type"] == "test"]
    
    validate_data_splits(data_list_train, data_list_validate, data_list_test)
    
    dataset_train = convert_data_list_to_dataset_(data_list_train)
    dataset_validate = convert_data_list_to_dataset_(data_list_validate)
    dataset_test = convert_data_list_to_dataset_(data_list_test)
    
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=dataset_train.collater,
        num_workers=0
    )
    data_loader_valid = DataLoader(
        dataset_validate, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=dataset_validate.collater,
        num_workers=0
    )
    data_loader_test = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=dataset_test.collater,
        num_workers=0
    )
    
    return data_loader_train, data_loader_valid, data_loader_test

# UniMol implementation based on:
# Zhou et al. "Uni-Mol: A Universal 3D Molecular Representation Learning Framework" ICLR 2023
class UniMolModel(nn.Module):
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('../data/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), 512, self.padding_idx)
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=15,
            embed_dim=512,
            ffn_embed_dim=2048,
            attention_heads=64,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            max_seq_len=512,
            activation_fn='gelu',
            no_final_head_layer_norm=True,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, 64, 'gelu'
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.apply(init_bert_params)

    def forward(self, sample):
        net_input = sample['input']
        src_tokens, src_distance, src_coord, src_edge_type = net_input['src_tokens'], net_input['src_distance'], \
                                                             net_input['src_coord'], net_input['src_edge_type']
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            _, _, _, _,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        output = {
            "molecule_embedding": encoder_rep,
            "molecule_representation": encoder_rep[:, 0, :],
            "smiles": sample['input']["smiles"],
        }
        return output


class UnimolMultitaskModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        pretrained_model_path = '../saved_models/pretrained/mol_pre_no_h_220816.pt'
        
        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model file does not exist: {pretrained_model_path}\n"
                                  f"Please ensure the file is in the correct location: {os.path.abspath(pretrained_model_path)}")
        
        self.molecule_encoder = UniMolModel()
        self.molecule_encoder.load_state_dict(th.load(pretrained_model_path, map_location=device)['model'], strict=False)
        for param in self.molecule_encoder.parameters():
            param.requires_grad = False
            
        self.feature_processor = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        self.output_CYP3A4_inhibitor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        self.output_CYP2C9_inhibitor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        
        self.output_F50 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        self.output_PPB = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        self.output_CYP3A4_substrate = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )
        self.output_CYP2D6_substrate = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )
        self.output_CLp_c = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )
        self.output_FDAMDD_c = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )
        self.output_DILI = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )
        self.output_Reproductive_toxicity = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )
        self.output_Ames = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )
        self.output_Pgp_inhibitor = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(256, 2)
        )
        
        self.output_Micronucleus = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
        self.output_BSEP_inhibitor = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
        self.output_Neurotoxicity = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
        
        self.output_VDss = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.to(device)

    def move_data_batch_to_cuda(self, data_batch, device):
        data_batch['input'] = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in data_batch['input'].items()}
        data_batch['target'] = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in data_batch['target'].items()}
        return data_batch

    def forward(self, data_batch):
        data_batch = self.move_data_batch_to_cuda(data_batch, next(self.parameters()).device)
        
        molecule_encoder_output = self.molecule_encoder(data_batch)
        molecule_embedding = molecule_encoder_output['molecule_embedding']
        
        batch_size = molecule_embedding.shape[0]
        features = self.feature_processor(molecule_embedding.view(-1, molecule_embedding.size(-1)))
        features = features.view(batch_size, -1, features.size(-1))
        
        attended_features, _ = self.attention(features, features, features)
        features = features + 0.1 * attended_features
        
        mean_features = features.mean(dim=1)
        max_features = features.max(dim=1)[0]
        pooled_features = (mean_features + max_features) / 2
        
        return {
            'F50': self.output_F50(pooled_features),
            'PPB': self.output_PPB(pooled_features),
            'CYP3A4_substrate': self.output_CYP3A4_substrate(pooled_features),
            'CYP3A4_inhibitor': self.output_CYP3A4_inhibitor(pooled_features),
            'CYP2D6_substrate': self.output_CYP2D6_substrate(pooled_features),
            'CYP2C9_inhibitor': self.output_CYP2C9_inhibitor(pooled_features),
            'CLp_c': self.output_CLp_c(pooled_features),
            'FDAMDD_c': self.output_FDAMDD_c(pooled_features),
            'DILI': self.output_DILI(pooled_features),
            'Micronucleus': self.output_Micronucleus(pooled_features),
            'Reproductive_toxicity': self.output_Reproductive_toxicity(pooled_features),
            'Ames': self.output_Ames(pooled_features),
            'BSEP_inhibitor': self.output_BSEP_inhibitor(pooled_features),
            'Pgp_inhibitor': self.output_Pgp_inhibitor(pooled_features),
            'Neurotoxicity': self.output_Neurotoxicity(pooled_features),
            'VDss': self.output_VDss(pooled_features).squeeze(-1),
        }


def evaluate(model, data_loader, csv_save, trial_version=None, data_dir=None):
    model.eval()
    device = next(model.parameters()).device
    
    classification_tasks = [
        'F50', 'PPB', 'CYP3A4_substrate', 'CYP3A4_inhibitor', 'CYP2D6_substrate',
        'CYP2C9_inhibitor', 'CLp_c', 'FDAMDD_c', 'DILI', 'Micronucleus',
        'Reproductive_toxicity', 'Ames', 'BSEP_inhibitor', 'Pgp_inhibitor', 'Neurotoxicity'
    ]
    regression_tasks = ['VDss']
    
    predictions = {task: [] for task in classification_tasks + regression_tasks}
    true_labels = {task: [] for task in classification_tasks + regression_tasks}
    masks = {task: [] for task in classification_tasks + regression_tasks}
    
    with th.no_grad():
        for data_batch in data_loader:
            data_batch = {k: {k2: v2.to(device) if isinstance(v2, th.Tensor) else v2 
                            for k2, v2 in v.items()} 
                        for k, v in data_batch.items()}
            
            outputs = model(data_batch)
            
            for task in classification_tasks + regression_tasks:
                task_pred = outputs[task].detach()
                task_true = data_batch['target'][task].detach()
                task_mask = data_batch['target'][f'{task}_mask'].detach()
                
                predictions[task].append(task_pred.cpu())
                true_labels[task].append(task_true.cpu())
                masks[task].append(task_mask.cpu())

    for task in predictions.keys():
        predictions[task] = th.cat(predictions[task], dim=0)
        true_labels[task] = th.cat(true_labels[task], dim=0)
        masks[task] = th.cat(masks[task], dim=0)

    metrics = {}
    
    total_correct = 0
    total_samples = 0
    
    for task in classification_tasks:
        valid_mask = masks[task].bool()
        if valid_mask.sum() == 0:
            continue
            
        task_pred = predictions[task][valid_mask]
        task_true = true_labels[task][valid_mask]
        
        probs = th.softmax(task_pred, dim=1).numpy()
        pred_labels = np.argmax(probs, axis=1)
        true_labels_np = task_true.numpy()
        
        task_metrics = {}
        
        try:
            task_metrics['accuracy'] = round(accuracy_score(true_labels_np, pred_labels), 3)
            task_metrics['balanced_accuracy'] = round(balanced_accuracy_score(true_labels_np, pred_labels), 3)
            
            task_metrics['auc_roc'] = round(roc_auc_score(true_labels_np, probs[:, 1]), 3)
            task_metrics['auc_prc'] = round(average_precision_score(true_labels_np, probs[:, 1]), 3)
            
            total_correct += (pred_labels == true_labels_np).sum()
            total_samples += len(true_labels_np)
            
            task_metrics['precision'] = round(precision_score(true_labels_np, pred_labels, average='binary'), 3)
            task_metrics['recall'] = round(recall_score(true_labels_np, pred_labels, average='binary'), 3)
            task_metrics['f1'] = round(2 * task_metrics['precision'] * task_metrics['recall'] / 
                                    (task_metrics['precision'] + task_metrics['recall'] + 1e-8), 3)
                
        except Exception as e:
            print(f"Error calculating metrics for task {task}: {str(e)}")
            continue
            
        metrics[task] = task_metrics
    
    regression_metrics = {}
    for task in regression_tasks:
        valid_mask = masks[task].bool()
        if valid_mask.sum() == 0:
            continue
            
        task_pred = predictions[task][valid_mask].numpy()
        task_true = true_labels[task][valid_mask].numpy()
        
        try:
            mse = np.mean((task_pred - task_true) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(task_pred - task_true))
            
            if np.var(task_true) == 0:
                r2 = 0.0
            else:
                r2 = 1 - (np.sum((task_true - task_pred) ** 2) / np.sum((task_true - np.mean(task_true)) ** 2))
            
            regression_metrics[f'{task}_mse'] = round(mse, 3)
            regression_metrics[f'{task}_rmse'] = round(rmse, 3)
            regression_metrics[f'{task}_mae'] = round(mae, 3)
            regression_metrics[f'{task}_r2'] = round(r2, 3)
            
        except Exception as e:
            print(f"Error calculating regression metrics for task {task}: {str(e)}")
            continue
    
    if regression_metrics:
        metrics['regression_metrics'] = regression_metrics

    if total_samples > 0:
        metrics['overall_accuracy'] = round(total_correct / total_samples, 3)

    if csv_save and trial_version is not None:
        evaluate_dir = f'{data_dir}/evaluate/trial_{trial_version}'
        os.makedirs(evaluate_dir, exist_ok=True)
        
        for task in classification_tasks:
            try:
                valid_mask = masks[task].bool()
                if valid_mask.sum() == 0:
                    continue
                    
                task_pred = predictions[task][valid_mask]
                task_true = true_labels[task][valid_mask]
                
                probs = th.softmax(task_pred, dim=1).numpy()
                pred_labels = np.argmax(probs, axis=1)
                true_labels_np = task_true.numpy()
                
                results_df = pd.DataFrame({
                    'true_label': true_labels_np,
                    'predicted_label': pred_labels,
                    'prob_class_0': probs[:, 0],
                    'prob_class_1': probs[:, 1]
                })
                
                save_path = f'{evaluate_dir}/predictions_{task}.csv'
                results_df.to_csv(save_path, index=False)
                print(f"Saved predictions for task {task} to {save_path}")
                
            except Exception as e:
                print(f"Error saving predictions for task {task}: {str(e)}")
        
        for task in regression_tasks:
            try:
                valid_mask = masks[task].bool()
                if valid_mask.sum() == 0:
                    continue
                    
                task_pred = predictions[task][valid_mask].numpy()
                task_true = true_labels[task][valid_mask].numpy()
                
                results_df = pd.DataFrame({
                    'true_value': task_true,
                    'predicted_value': task_pred,
                    'absolute_error': np.abs(task_true - task_pred)
                })
                
                save_path = f'{evaluate_dir}/predictions_{task}.csv'
                results_df.to_csv(save_path, index=False)
                print(f"Saved predictions for task {task} to {save_path}")
                
            except Exception as e:
                print(f"Error saving predictions for task {task}: {str(e)}")

    return metrics


class EarlyStopping:
    def __init__(self, patience=100, verbose=True, delta=1e-4, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.inf
        self.delta = delta
        self.path = path
        
    def __call__(self, metrics, model):
        classification_score = 0
        classification_count = 0
        

        for task, task_metrics in metrics.items():
            if task != 'regression_metrics' and task != 'overall_accuracy' and isinstance(task_metrics, dict):
                if 'auc_roc' in task_metrics:
                    classification_score += task_metrics['auc_roc']
                    classification_count += 1
                elif 'accuracy' in task_metrics:
                    classification_score += task_metrics['accuracy']
                    classification_count += 1
        
        classification_avg = classification_score / max(classification_count, 1)
        
        regression_score = 0
        regression_count = 0
        
        if 'regression_metrics' in metrics:
            reg_metrics = metrics['regression_metrics']
            for metric_name, value in reg_metrics.items():
                if metric_name.endswith('_r2'):
                    regression_score += value
                    regression_count += 1
                elif metric_name.endswith('_rmse') and not any(m.endswith('_r2') for m in reg_metrics.keys()):
                    regression_score += -value
                    regression_count += 1
        
        regression_avg = regression_score / max(regression_count, 1) if regression_count > 0 else 0
        
        if regression_count > 0:
            score = (classification_avg + regression_avg) / 2
        else:
            score = classification_avg
            
        if 'overall_accuracy' in metrics:
            score = 0.7 * score + 0.3 * metrics['overall_accuracy']
        
        if self.verbose:
            print(f"Validation score: {score:.4f} (Classification: {classification_avg:.4f}, Regression: {regression_avg:.4f})")
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return True
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            return True
            
    def save_checkpoint(self, model):
        if self.verbose:
            print(f'Validation score improved ({self.val_score_max:.6f} --> {self.best_score:.6f}). Saving model...')
        th.save(model.state_dict(), self.path)
        self.val_score_max = self.best_score

def train(trial_version, device, data_dir):
    wandb.init(
        project="MUTITASK_UNIMOL_FINAL", 
        name=f"trial_{trial_version}",
        config={
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "batch_size": 6,
            "epochs": 2000,
            "patience": 100,
            "device": device,
        }
    )

    data_loader_train, data_loader_validate, data_loader_test = convert_data_list_to_data_loader(data_dir)

    model = UnimolMultitaskModel(device)

    classification_tasks = [
        'F50', 'PPB', 'CYP3A4_substrate', 'CYP3A4_inhibitor', 'CYP2D6_substrate',
        'CYP2C9_inhibitor', 'CLp_c', 'FDAMDD_c', 'DILI', 'Micronucleus',
        'Reproductive_toxicity', 'Ames', 'BSEP_inhibitor', 'Pgp_inhibitor', 'Neurotoxicity'
    ]
    regression_tasks = ['VDss']
    all_tasks = classification_tasks + regression_tasks
    
    classification_loss = nn.CrossEntropyLoss(reduction='none')
    regression_loss = nn.MSELoss(reduction='none')

    optimizer = optim.Adam(
        model.parameters(),
        lr=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=40,
        T_mult=2,
        eta_min=1e-8
    )

    early_stopping = EarlyStopping(
        patience=wandb.config.patience,
        verbose=True,
        delta=1e-4,
        path=f'../saved_models/trained/{trial_version}_best.pt'
    )
    device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
    
    scaler = GradScaler() if device_type == 'cuda' else None

    for epoch in range(wandb.config.epochs):
        model.train()
        epoch_losses = {task: 0.0 for task in all_tasks}
        epoch_samples = {task: 0 for task in all_tasks}
        
        for step, data_batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            
            with autocast(device_type=device_type, dtype=th.float16 if device_type == 'cuda' else th.float32):
                label_predict_batch = model(data_batch)
                
                valid_task_count = 0
                task_losses = {}
                
                for task in classification_tasks:
                    try:
                        label_value = data_batch['target'][task].to(device)
                        label_mask = data_batch['target'][f'{task}_mask'].to(device)
                        
                        valid_indices = label_mask.bool()
                        if not valid_indices.any():
                            continue
                        
                        valid_predictions = label_predict_batch[task][valid_indices]
                        valid_labels = label_value[valid_indices]
                        
                        loss_per_sample = classification_loss(valid_predictions, valid_labels.long())
                        current_loss = loss_per_sample.mean()
                        
                        if not th.isnan(current_loss) and not th.isinf(current_loss):
                            task_losses[task] = current_loss
                            valid_task_count += 1
                            epoch_losses[task] += current_loss.item()
                            epoch_samples[task] += valid_indices.sum().item()
                            
                    except Exception as e:
                        print(f"Error in classification task {task}: {str(e)}")
                        continue
                
                for task in regression_tasks:
                    try:
                        label_value = data_batch['target'][task].to(device)
                        label_mask = data_batch['target'][f'{task}_mask'].to(device)
                        
                        valid_indices = label_mask.bool()
                        if not valid_indices.any():
                            continue
                        
                        valid_predictions = label_predict_batch[task][valid_indices]
                        valid_labels = label_value[valid_indices]
                        
                        loss_per_sample = regression_loss(valid_predictions, valid_labels)
                        current_loss = loss_per_sample.mean()
                        
                        if not th.isnan(current_loss) and not th.isinf(current_loss):
                            task_losses[task] = current_loss
                            valid_task_count += 1
                            epoch_losses[task] += current_loss.item()
                            epoch_samples[task] += valid_indices.sum().item()
                            
                    except Exception as e:
                        print(f"Error in regression task {task}: {str(e)}")
                        continue

                loss = sum(task_losses.values()) / max(valid_task_count, 1) if valid_task_count > 0 else 0

            if loss > 0:
                if device_type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            if (step + 1) % 19 == 0:
                loss_info = {task: epoch_losses[task]/max(epoch_samples[task], 1) 
                           for task in all_tasks}
                print(f'Epoch: {epoch}, Step: {step}')
                print('Task losses:', {k: round(v, 3) for k, v in loss_info.items()})
                print('Total loss:', round(loss.item(), 3))
                
                wandb.log({
                    f"train_loss_{k}": v/max(epoch_samples[k], 1) 
                    for k, v in epoch_losses.items()
                })
                wandb.log({"train_total_loss": loss.item()})

        scheduler.step()

        metric_train = evaluate(model, data_loader_train, csv_save=False, trial_version=trial_version, data_dir=data_dir)
        metric_validate = evaluate(model, data_loader_validate, csv_save=False, trial_version=trial_version, data_dir=data_dir)
        metric_test = evaluate(model, data_loader_test, csv_save=False, trial_version=trial_version, data_dir=data_dir)

        print("==================================================================================")
        print('Epoch', epoch)
        print('Train', metric_train)
        print('validate', metric_validate) 
        print('Test', metric_test)
        print("==================================================================================")

        wandb.log({
            "epoch": epoch,
            **{f"train_{task}_{metric}": value 
               for task, metrics in metric_train.items() 
               if isinstance(metrics, dict)
               for metric, value in metrics.items()},
            **{f"val_{task}_{metric}": value 
               for task, metrics in metric_validate.items() 
               if isinstance(metrics, dict)
               for metric, value in metrics.items()},
            **{f"test_{task}_{metric}": value 
               for task, metrics in metric_test.items() 
               if isinstance(metrics, dict)
               for metric, value in metrics.items()}
        })

        if 'overall_accuracy' in metric_train:
            wandb.log({"train_overall_accuracy": metric_train['overall_accuracy']})
        if 'overall_accuracy' in metric_validate:
            wandb.log({"val_overall_accuracy": metric_validate['overall_accuracy']})
        if 'overall_accuracy' in metric_test:
            wandb.log({"test_overall_accuracy": metric_test['overall_accuracy']})
            
        if 'regression_metrics' in metric_train:
            for metric, value in metric_train['regression_metrics'].items():
                wandb.log({f"train_regression_{metric}": value})
        if 'regression_metrics' in metric_validate:
            for metric, value in metric_validate['regression_metrics'].items():
                wandb.log({f"val_regression_{metric}": value})
        if 'regression_metrics' in metric_test:
            for metric, value in metric_test['regression_metrics'].items():
                wandb.log({f"test_regression_{metric}": value})

        is_best = early_stopping(metric_validate, model)
        
        if is_best:
            best_results_dir = f'{data_dir}/evaluate/trial_{trial_version}'
            os.makedirs(best_results_dir, exist_ok=True)
            csv_path = f'{best_results_dir}/best_results.csv'
            
            best_metrics = {
                'epoch': epoch,
                'best_epoch': epoch,
                'val_score': early_stopping.best_score,
                'metrics_train': metric_train,
                'metrics_validate': metric_validate,
                'metrics_test': metric_test
            }
            
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                header = ['epoch', 'best_epoch', 'val_score']
                
                for split in ['train', 'validate', 'test']:
                    for task in classification_tasks:
                        if task in best_metrics[f'metrics_{split}']:
                            for metric_name in best_metrics[f'metrics_{split}'][task]:
                                header.append(f"{split}_{metric_name}_{task}")
                
                for split in ['train', 'validate', 'test']:
                    if 'regression_metrics' in best_metrics[f'metrics_{split}']:
                        for metric_name in best_metrics[f'metrics_{split}']['regression_metrics']:
                            for task in regression_tasks:
                                header.append(f"{split}_{metric_name}_{task}")
                
                writer.writerow(header)
                
                row = [
                    best_metrics['epoch'],
                    best_metrics['best_epoch'],
                    best_metrics['val_score']
                ]
                
                for split in ['train', 'validate', 'test']:
                    for task in classification_tasks:
                        if task in best_metrics[f'metrics_{split}']:
                            for metric_name in best_metrics[f'metrics_{split}'][task]:
                                row.append(best_metrics[f'metrics_{split}'][task][metric_name])
                
                for split in ['train', 'validate', 'test']:
                    if 'regression_metrics' in best_metrics[f'metrics_{split}']:
                        for metric_name in best_metrics[f'metrics_{split}']['regression_metrics']:
                            for task in regression_tasks:
                                row.append(best_metrics[f'metrics_{split}']['regression_metrics'][metric_name])
                
                writer.writerow(row)
                
            print(f"Best results saved to {csv_path}")

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    wandb.finish()


def test(trial_version, device, data_dir):
    model = UnimolMultitaskModel(device)
    
    model_path = f'../saved_models/trained/{trial_version}_best.pt'
    if os.path.exists(model_path):
        model.load_state_dict(th.load(model_path, weights_only=True))
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    _, _, data_loader_test = convert_data_list_to_data_loader(data_dir)
    
    evaluate_dir = f'{data_dir}/evaluate/trial_{trial_version}'
    os.makedirs(evaluate_dir, exist_ok=True)
    
    print("\nGenerating evaluation results...")
    metrics = evaluate(
        model=model,
        data_loader=data_loader_test,
        csv_save=True,
        trial_version=trial_version,
        data_dir=data_dir
    )
    
    classification_tasks = [
        'F50', 'PPB', 'CYP3A4_substrate', 'CYP3A4_inhibitor', 'CYP2D6_substrate',
        'CYP2C9_inhibitor', 'CLp_c', 'FDAMDD_c', 'DILI', 'Micronucleus',
        'Reproductive_toxicity', 'Ames', 'BSEP_inhibitor', 'Pgp_inhibitor', 'Neurotoxicity'
    ]
    regression_tasks = ['VDss']
    
    print("\nTest Results:")
    print("-" * 50)
    
    print("\nClassification Tasks:")
    for task in classification_tasks:
        if task in metrics:
            print(f"\n{task}:")
            task_metrics = metrics[task]
            for metric_name, value in task_metrics.items():
                print(f"  {metric_name}: {value:.3f}")
    
    if 'regression_metrics' in metrics:
        print("\nRegression Tasks:")
        for task in regression_tasks:
            print(f"\n{task}:")
            for metric_name, value in metrics['regression_metrics'].items():
                if metric_name.startswith(f'{task}_'):
                    print(f"  {metric_name[len(task)+1:]}: {value:.3f}")
    
    if 'overall_accuracy' in metrics:
        print(f"\nOverall Classification Accuracy: {metrics['overall_accuracy']:.3f}")
    
    print("-" * 50)
    
    summary_path = f'{evaluate_dir}/test_summary.csv'
    
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    try:
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Metric', 'Value'])
            
            for task in classification_tasks:
                if task in metrics and isinstance(metrics[task], dict):
                    for metric_name, value in metrics[task].items():
                        writer.writerow([task, metric_name, f"{value:.3f}"])
            
            if 'regression_metrics' in metrics:
                for metric_name, value in metrics['regression_metrics'].items():
                    task = metric_name.split('_')[0]
                    metric = '_'.join(metric_name.split('_')[1:])
                    writer.writerow([task, metric, f"{value:.3f}"])
            
            if 'overall_accuracy' in metrics:
                writer.writerow(['Overall', 'accuracy', f"{metrics['overall_accuracy']:.3f}"])
        
        print(f"\nDetailed predictions saved in: {evaluate_dir}/")
        print(f"Summary results saved to: {summary_path}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Training script with configurable seed directory')
    parser.add_argument('--seed_dir', type=str, default='merged_seed_8',
                       help='Name of the seed directory under ../data/')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--trial_version', type=str, default='1',
                       help='Trial version number')
    args = parser.parse_args()
    
    data_dir = f'../data/{args.seed_dir}'
    csv_path = f'{data_dir}/{args.seed_dir}.csv'
    
    os.makedirs(data_dir, exist_ok=True)
    
    device = th.device(args.device)
    print(f"Using device: {device}")
    
    set_random_seed(1024)
    print(f"Processing data from {csv_path}")
    print("Data processing start!") 
    calculate_molecule_3D_structure(csv_path, data_dir)
    construct_data_list(data_dir)

    try:
        print("Training start!")
        train(trial_version=args.trial_version, device=device, data_dir=data_dir)
        
        print("Testing start!")
        test(trial_version=args.trial_version, device=device, data_dir=data_dir)
        
        print('All is well!')
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise