import os
import torch as th
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle as pkl
from model import UnimolMultitaskModel, calculate_molecule_3D_structure, convert_data_list_to_data_loader
from unicore.data import Dictionary

class MoleculePredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.dictionary = Dictionary.load('../data/token_list.txt')
        self.dictionary.add_symbol("[MASK]", is_special=True)
        self.model = UnimolMultitaskModel()
        self.model.load_state_dict(th.load(model_path))
        self.model.to(device)
        self.model.eval()
        self.task_labels = {
            'CLp_label': ['High', 'Medium', 'Low'],
            'T50_label': ['Very Low', 'Low', 'Medium', 'High'],
            'CYP2C19_inhibitor': ['Non-inhibitor', 'Inhibitor'],
            'CYP2C9_inhibitor': ['Non-inhibitor', 'Inhibitor'],
            'CYP3A4_inhibitor': ['Non-inhibitor', 'Inhibitor'],
            'CYP3A4_substrate': ['Non-substrate', 'Substrate'],
            'MDCK': ['Low', 'High'],
            'Neurotoxicity': ['Non-toxic', 'Toxic'],
            'Skin_corrosion': ['Non-corrosive', 'Corrosive']
        }

    def prepare_single_molecule(self, smiles):
        "Process a single SMILES string"
        th.manual_seed(1024)
        np.random.seed(1024)
        temp_dir = 'temp_prediction'
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(f'{temp_dir}/intermediate', exist_ok=True)
        for f in ['invalid_smiles.txt', 'smiles_to_conformation_dict.pkl', 'data_list.pkl']:
            path = os.path.join(temp_dir, 'intermediate', f)
            if os.path.exists(path):
                os.remove(path)
        temp_df = pd.DataFrame({
            'smiles': [smiles],
            'sequence': ['temp'],
            'label': [0],
            'dataset_type': ['test'],
            'CLp_label': [-1],
            'T50_label': [-1],
            'CYP2C19_inhibitor': [-1],
            'CYP2C9_inhibitor': [-1],
            'CYP3A4_inhibitor': [-1],
            'CYP3A4_substrate': [-1],
            'MDCK': [-1],
            'Neurotoxicity': [-1],
            'Skin_corrosion': [-1]
        })
        temp_csv = f'{temp_dir}/temp.csv'
        temp_df.to_csv(temp_csv, index=False)
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                return False
            molecule = AllChem.AddHs(molecule)
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            result = AllChem.EmbedMolecule(molecule, randomSeed=42, useRandomCoords=True, maxAttempts=1000)
            if result != 0:
                return False
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                return False
            coordinates = molecule.GetConformer().GetPositions()
            coordinate_list = [coordinates.astype(np.float32)]
            smiles_to_conformation_dict = {
                smiles: {
                    'smiles': smiles,
                    'atoms': atoms,
                    'coordinates': coordinate_list
                }
            }
            pkl.dump(smiles_to_conformation_dict, open(f'{temp_dir}/intermediate/smiles_to_conformation_dict.pkl', 'wb'))
            data_item = {
                "atoms": atoms,
                "coordinates": coordinate_list,
                "smiles": smiles,
                "sequence": "temp",
                "label": 0,
                "dataset_type": "test"
            }
            pkl.dump([data_item], open(f'{temp_dir}/intermediate/data_list.pkl', 'wb'))
            return True
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {str(e)}")
            return False

    def predict_smiles(self, smiles):
        "Predict the result for a single SMILES"
        if not self.prepare_single_molecule(smiles):
            return {"error": "Failed to process SMILES structure"}
        try:
            _, _, data_loader = convert_data_list_to_data_loader()
            self.model.eval()
            results = {}
            with th.no_grad():
                for batch in data_loader:
                    batch = {
                        'input': {k: v.to(self.device) if isinstance(v, th.Tensor) else v for k, v in batch['input'].items()},
                        'target': {k: v.to(self.device) if isinstance(v, th.Tensor) else v for k, v in batch['target'].items()}
                    }
                    outputs = self.model(batch)
                    for task, pred in outputs.items():
                        probs = th.softmax(pred, dim=1)[0]
                        pred_class = th.argmax(probs).item()
                        results[task] = {
                            'predicted_class': self.task_labels[task][pred_class],
                            'probabilities': {label: float(prob) for label, prob in zip(self.task_labels[task], probs.cpu().numpy())}
                        }
            return results
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def predict_csv(self, csv_path, output_path=None):
        "Predict all SMILES in CSV file"
        df = pd.read_csv(csv_path)
        if 'smiles' not in df.columns:
            raise ValueError("CSV must contain 'smiles' column")
        all_results = []
        for idx, row in df.iterrows():
            smiles = row['smiles']
            try:
                result = self.predict_smiles(smiles)
                result['smiles'] = smiles
                all_results.append(result)
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {str(e)}")
                continue
        results_df = pd.DataFrame(all_results)
        if output_path:
            results_df.to_csv(output_path, index=False)
        return results_df

def main():
    predictor = MoleculePredictor(
        model_path='saved_models/trained/Multitask_Unimol_best.pt',
        device='cuda' if th.cuda.is_available() else 'cpu'
    )
    smiles = "CC(=O)OC1=CC=CC:C1C(=O)O"
    result = predictor.predict_smiles(smiles)
    print("\nPrediction for single SMILES:")
    print(f"SMILES: {smiles}")
    print("\nPrediction results:")
    for task, pred in result.items():
        print(f"\n{task}:")
        print(f"Predicted class: {pred['predicted_class']}")
        print("Label prediction probabilities:")
        for label, prob in pred['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
    csv_path = "test_molecules.csv"
    if os.path.exists(csv_path):
        print("\nPredicting molecules from CSV file...")
        results_df = predictor.predict_csv(csv_path=csv_path, output_path="prediction_results.csv")
        print("\nPrediction results saved to prediction_results.csv")

if __name__ == "__main__":
    main()