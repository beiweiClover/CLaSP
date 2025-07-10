#!/usr/bin/env python
# coding: utf-8
"""
predict_single_smiles.py

Usage:
python predict_single_smiles.py \
    --smiles "CCO" \
    --checkpoint ../saved_models/trained/1_best.pt \
    --device cuda
"""
import os
import json
import argparse
import pickle as pkl
import numpy as np
import torch as th
from torch.utils.data import DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem

# ---------- 你训练脚本里已有的依赖 ----------
from unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset,
    PrependTokenDataset, AppendTokenDataset, FromNumpyDataset,
    RightPadDataset, RightPadDataset2D, RawArrayDataset
)
from unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord,
)
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer
# ------------------------------------------------

# ------------------- 随机数固定 -------------------
def set_random_seed(random_seed=1024):
    import random, torch, numpy as np, os
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# -------------------------------------------------

# ---------------- UniMol 多任务模型 ----------------
class UniMolModel(th.nn.Module):
    """与训练阶段保持完全一致（删去与推理无关的注释）"""
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('../data/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = th.nn.Embedding(len(dictionary), 512, self.padding_idx)

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
        self.gbf_proj = NonLinearHead(K, 64, 'gelu')
        self.gbf = GaussianLayer(K, n_edge_type)

        # 初始化权重
        from unicore.modules import init_bert_params
        self.apply(init_bert_params)

    def _graph_bias(self, dist, et):
        n = dist.size(-1)
        gbf_feature = self.gbf(dist, et)
        gbf_result = self.gbf_proj(gbf_feature)              # (B, N, N, K)->(B, K, N, N)
        gbf_result = gbf_result.permute(0, 3, 1, 2).contiguous()
        return gbf_result.view(-1, n, n)

    def forward(self, sample):
        x = self.embed_tokens(sample['input']['src_tokens'])
        graph_bias = self._graph_bias(sample['input']['src_distance'],
                                      sample['input']['src_edge_type'])
        encoder_rep, *_ = self.encoder(
            x,
            padding_mask=sample['input']['src_tokens'].eq(self.padding_idx),
            attn_mask=graph_bias
        )
        return {
            "molecule_embedding": encoder_rep,
            "molecule_representation": encoder_rep[:, 0, :],  # cls token
        }

class UnimolMultitaskModel(th.nn.Module):
    """同训练一致（保留推理所需部分）"""
    def __init__(self, device):
        super().__init__()
        self.molecule_encoder = UniMolModel()
        # 加载你的预训练（冻结）
        pre_path = '../saved_models/pretrained/mol_pre_no_h_220816.pt'
        self.molecule_encoder.load_state_dict(
            th.load(pre_path, map_location=device)['model'], strict=False)
        for p in self.molecule_encoder.parameters():
            p.requires_grad = False

        # 后续特征处理 & 输出头
        self.feature_processor = th.nn.Sequential(
            th.nn.Linear(512, 256),
            th.nn.BatchNorm1d(256),
            th.nn.GELU(),
            th.nn.Dropout(0.3),
            th.nn.Linear(256, 256),
            th.nn.BatchNorm1d(256),
        )
        self.attention = th.nn.MultiheadAttention(256, 4, dropout=0.2, batch_first=True)

        # 任务列表
        self.class_tasks = [
            'F50', 'PPB', 'CYP3A4_substrate', 'CYP3A4_inhibitor', 'CYP2D6_substrate',
            'CYP2C9_inhibitor', 'CLp_c', 'FDAMDD_c', 'DILI', 'Micronucleus',
            'Reproductive_toxicity', 'Ames', 'BSEP_inhibitor', 'Pgp_inhibitor', 'Neurotoxicity'
        ]
        self.reg_tasks = ['VDss']

        # 按数据量分级 Dropout，同训练
        def make_head(drop_p, out_dim): return th.nn.Sequential(
            th.nn.Dropout(drop_p), th.nn.Linear(256, out_dim)
        )
        # 以下与训练代码保持相同 Dropout 设置
        self.output_heads = th.nn.ModuleDict({
            'CYP3A4_inhibitor': make_head(0.2, 2),
            'CYP2C9_inhibitor': make_head(0.2, 2),

            'F50': make_head(0.3, 2),
            'PPB': make_head(0.3, 2),
            'CYP3A4_substrate': make_head(0.25, 2),
            'CYP2D6_substrate': make_head(0.25, 2),
            'CLp_c': make_head(0.25, 2),
            'FDAMDD_c': make_head(0.25, 2),
            'DILI': make_head(0.25, 2),
            'Reproductive_toxicity': make_head(0.25, 2),
            'Ames': make_head(0.25, 2),
            'Pgp_inhibitor': make_head(0.25, 2),

            'Micronucleus': make_head(0.4, 2),
            'BSEP_inhibitor': make_head(0.4, 2),
            'Neurotoxicity': make_head(0.4, 2),

            'VDss': make_head(0.3, 1),
        })

        self.to(device)

    def forward(self, batch):
        enc = self.molecule_encoder(batch)['molecule_embedding']
        B = enc.size(0)
        feat = self.feature_processor(enc.view(-1, enc.size(-1))).view(B, -1, 256)
        attn_out, _ = self.attention(feat, feat, feat)
        feat = feat + 0.1 * attn_out
        pooled = (feat.mean(dim=1) + feat.max(dim=1)[0]) / 2

        out = {}
        for t in self.class_tasks:
            out[t] = self.output_heads[t](pooled)
        out['VDss'] = self.output_heads['VDss'](pooled).squeeze(-1)
        return out
# -------------------------------------------------

# --------- 单条 SMILES → atoms + 3D 坐标 ----------
def smiles_to_atoms_coords(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(
        mol, randomSeed=42, useRandomCoords=True, maxAttempts=1000
    )
    if res != 0:
        raise RuntimeError(f"EmbedMolecule failed ({res}): {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    coords = mol.GetConformer().GetPositions()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return atoms, coords.astype(np.float32)
# -------------------------------------------------

# ---- 把数据包装成与你训练时同格式的 NestedDataset ----
def build_dataloader_from_smiles(smiles_list, batch_size=1):
    # 1. 把 SMILES 转 data_list
    label_defaults = {
        'F50': -1, 'PPB': -1, 'CYP3A4_substrate': -1, 'CYP3A4_inhibitor': -1,
        'CYP2D6_substrate': -1, 'CYP2C9_inhibitor': -1, 'CLp_c': -1,
        'FDAMDD_c': -1, 'DILI': -1, 'Micronucleus': -1,
        'Reproductive_toxicity': -1, 'Ames': -1, 'BSEP_inhibitor': -1,
        'Pgp_inhibitor': -1, 'Neurotoxicity': -1, 'VDss': float('nan')
    }
    data_list = []
    for smi in smiles_list:
        atoms, coords = smiles_to_atoms_coords(smi)
        item = dict(
            atoms=atoms,
            coordinates=[coords],
            smiles=smi,
            dataset_type='test'
        )
        # 填充占位标签 & mask=0
        for lab, v in label_defaults.items():
            item[lab] = v
            item[f"{lab}_mask"] = 0.0
        data_list.append(item)

    # 2. 复用训练同名函数: token 化、坐标、边类型……
    dictionary = Dictionary.load('../data/token_list.txt')
    dictionary.add_symbol("[MASK]", is_special=True)

    smiles_ds = KeyDataset(data_list, "smiles")

    label_ds, mask_ds = {}, {}
    for lab in label_defaults:
        label_ds[lab] = th.utils.data.Dataset()  # 不会用到, 给接口占位
        mask_ds[f"{lab}_mask"] = th.utils.data.Dataset()

    base = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
    base = AtomTypeDataset(data_list, base)
    base = RemoveHydrogenDataset(base, "atoms", "coordinates", True, False)
    base = CroppingDataset(base, 1, "atoms", "coordinates", 256)
    base = NormalizeDataset(base, "coordinates", True)

    tok_ds = TokenizeDataset(KeyDataset(base, "atoms"), dictionary, 512)
    coord_ds = KeyDataset(base, "coordinates")
    coord_ds = AppendTokenDataset(PrependTokenDataset(FromNumpyDataset(coord_ds), 0.0), 0.0)
    src_tokens = AppendTokenDataset(PrependTokenDataset(tok_ds, dictionary.bos()), dictionary.eos())
    edge_type = EdgeTypeDataset(src_tokens, len(dictionary))
    distance = DistanceDataset(coord_ds)

    dataset = NestedDictionaryDataset({
        "input": {
            "src_tokens": RightPadDataset(src_tokens, dictionary.pad()),
            "src_coord": RightPadDatasetCoord(coord_ds, pad_idx=0),
            "src_distance": RightPadDataset2D(distance, pad_idx=0),
            "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0),
            "smiles": RawArrayDataset(smiles_ds),
        },
        "target": {}   # 预测阶段不需要标签
    })

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=dataset.collater, num_workers=0)
# -------------------------------------------------

# ---------------------- 主流程 --------------------
def predict_single_smiles(smiles: str, ckpt: str, device: str):
    set_random_seed(1024)
    device = th.device(device)
    model = UnimolMultitaskModel(device)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(th.load(ckpt, map_location=device), strict=False)
    model.eval()

    loader = build_dataloader_from_smiles([smiles])

    all_out = []
    for _ in range(3):                # 连续跑 3 次
        for batch in loader:
            batch = {
                'input': {k: v.to(device) if isinstance(v, th.Tensor) else v
                          for k, v in batch['input'].items()}
            }
            with th.no_grad():
                out = model(batch)
            # --------- 结果后处理 ----------
            pred = {}
            for t in model.class_tasks:
                prob = th.softmax(out[t], dim=-1).cpu().numpy()[0]
                pred[t] = {
                    "prob": prob.tolist(),
                    "pred_label": int(prob.argmax())
                }
            pred['VDss'] = float(out['VDss'].cpu().numpy()[0])
            all_out.append(pred)
    return all_out
# -------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, required=True,
                        help='SMILES string to predict')
    parser.add_argument('--checkpoint', type=str,
                        default='../saved_models/trained/seed_5818_V3_best.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    results = predict_single_smiles(
        smiles=args.smiles,
        ckpt=args.checkpoint,
        device=args.device
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
