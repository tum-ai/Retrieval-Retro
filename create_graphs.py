import os 
import re
import ast
import json
import pandas as pd
import numpy as np
import torch 
from torch_geometric.data import Data
from pymatgen.core import Composition
from typing import Dict, List, Set, Tuple, Optional
import argparse

class MaterialDatasetProcessor:
    def __init__(self, matscholar_path: str):
        """
        Initialize the dataset processor.
        
        Args:
            matscholar_path: Path to matscholar embeddings file
        """
        self.matscholar_embeddings = self._load_matscholar_embeddings(matscholar_path)
        self.element_lookup = None
        self.precursor_lookup = None
        
    def _load_matscholar_embeddings(self, path: str) -> Dict[str, torch.Tensor]:
        """Load and process matscholar embeddings."""
        df_matscholar = pd.read_csv(path, index_col=0)
        return {
            element: torch.tensor(row.values, dtype=torch.float32)
            for element, row in df_matscholar.iterrows()
        }
    
    def prepare_dataset_lookups(self, train_path: str, use_formation_energy: bool = False, val_path: Optional[str] = None, 
                              test_path: Optional[str] = None) -> None:
        """
        Create consistent lookups from all datasets.
        
        Args:
            train_path: Path to training data
            val_path: Optional path to validation data
            test_path: Optional path to test data
            use_formation_energy: Whether this is for formation energy prediction (no precursors)
        """
        # Get unique elements across all datasets
        all_elements = set()
        all_precursors = set()
        
        for path in [p for p in [train_path, val_path, test_path] if p]:
            elements, precursors = self._extract_unique_elements_and_precursors(path, use_formation_energy)
            all_elements.update(elements)
            if not use_formation_energy:
                all_precursors.update(precursors)
        
        # Create sorted lookups for consistency
        self.element_lookup = np.array(sorted(all_elements))
        if not use_formation_energy:
            self.precursor_lookup = np.array(sorted(all_precursors))
        else:
            self.precursor_lookup = np.array([])
        
        print(f"Found {len(self.element_lookup)} unique elements")
        if not use_formation_energy:
            print(f"Found {len(self.precursor_lookup)} unique precursors")
    
    def _extract_unique_elements_and_precursors(self, csv_path: str, use_formation_energy: bool = False) -> Tuple[Set[str], Set[str]]:
        """Extract unique elements and precursors from a dataset."""
        df = pd.read_csv(csv_path)
        
        # Process target materials
        unique_elements = set()
        for material in df['target_formula']:
            try:
                comp = Composition(material).element_composition
                unique_elements.update(comp.get_el_amt_dict().keys())
            except Exception as e:
                print(f"Error processing material {material}: {e}")
        
        # Process precursors only if not doing formation energy prediction
        unique_precursors = set()
        if not use_formation_energy and 'precursor_formulas' in df.columns:
            for precursor_set in df['precursor_formulas']:
                try:
                    formulas = ast.literal_eval(precursor_set)
                    unique_precursors.update(formulas)
                except Exception as e:
                    print(f"Error processing precursors {precursor_set}: {e}")
        
        return unique_elements, unique_precursors
    
    def _process_material_composition(self, material: str) -> List[Tuple[str, float]]:
        """Process a single material's composition."""
        try:
            fractional_comp = Composition(material).fractional_composition
            return list(fractional_comp.get_el_amt_dict().items())
        except Exception as e:
            print(f"Error processing material {material}: {e}")
            return []
    
    def create_graph_dataset(self, csv_path: str, dataset_type: str, 
                           output_dir: str, use_formation_energy: bool = False) -> List[Data]:
        """
        Create graph dataset from input CSV file.
        
        Args:
            csv_path: Path to input CSV file
            dataset_type: Type of dataset ('train', 'val', or 'test')
            output_dir: Directory to save the processed dataset
            use_formation_energy: Whether to use formation energy for the target material
        """
        if self.element_lookup is None:
            raise ValueError("Must call prepare_dataset_lookups before creating graphs")
        if not use_formation_energy and self.precursor_lookup is None:
            raise ValueError("Must call prepare_dataset_lookups before creating graphs when not using formation energy")
        
        df = pd.read_csv(csv_path)
        
        if not use_formation_energy:
            # Group by target_formula to get all precursor sets for each target
            grouped_df = df.groupby('target_formula')['precursor_formulas'].agg(list).reset_index()
        else:
            grouped_df = df

        data_list = []
        
        for idx, row in grouped_df.iterrows():
            try:
                # Process target material
                comp_tuples = self._process_material_composition(row['target_formula'])
                if not comp_tuples:
                    continue
                
                # Create node features
                x_list = []
                comp_fea = torch.zeros(len(self.element_lookup))
                for element, amount in comp_tuples:
                    x_list.append(self.matscholar_embeddings[element])
                    element_idx = np.where(self.element_lookup == element)[0][0]
                    comp_fea[element_idx] = amount
                
                x = torch.stack(x_list)
                fc_weight = comp_fea[comp_fea.nonzero()].reshape(-1)
                
                # Create edge features
                edge_index = []
                nodes = torch.arange(len(x_list))
                for n1 in nodes:
                    for n2 in nodes:
                        edge_index.append([n1.item(), n2.item()])
                edge_index = torch.tensor(edge_index).t()
                edge_attr = torch.rand(edge_index.shape[1], 400)
                
                if not use_formation_energy:
                    # Process all precursor sets for this target
                    all_precursor_sets = [ast.literal_eval(p_set) for p_set in row['precursor_formulas']]
                    num_precursor_sets = len(all_precursor_sets)
                    
                    # Create target labels - stack multiple precursor set embeddings
                    y_multiple = torch.zeros(num_precursor_sets, len(self.precursor_lookup))
                    for i, precursor_set in enumerate(all_precursor_sets):
                        for precursor in precursor_set:
                            precursor_idx = np.where(self.precursor_lookup == precursor)[0][0]
                            y_multiple[i, precursor_idx] = 1
                
                    # Calculate y_lb_all using OR operation across all combinations
                    y_lb_all = torch.any(y_multiple, dim=0).float()
                
                    # Create a separate graph for each precursor combination
                    data_list_for_target = []
                    for i in range(num_precursor_sets):
                        data = Data(
                            x=x,
                            edge_index=edge_index, 
                            edge_attr=edge_attr,
                            fc_weight=fc_weight,
                            comp_fea=comp_fea,
                            y_multiple=y_multiple,
                            y_multiple_len=torch.tensor(num_precursor_sets),
                            y_lb_freq=y_lb_all,
                            y_lb_avg=y_lb_all,
                            y_lb_all=y_lb_all,
                            y_lb_one=y_multiple[i],  # Each combination becomes y_lb_one for a separate graph
                            y_string_label=row['target_formula']
                        )
                        data_list_for_target.append(data)
                    data_list.extend(data_list_for_target)
                else:
                    data = Data(
                        x=x,
                        edge_index=edge_index, 
                        edge_attr=edge_attr,
                        fc_weight=fc_weight,
                        comp_fea=comp_fea,
                        y_exp_form=row['formation_energy']
                    )
                    data_list.append(data)
                
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        # Save dataset
        output_path = os.path.join(output_dir, f"mit_impact_dataset_{dataset_type}.pt") if not use_formation_energy else os.path.join(output_dir, f"mit_impact_dataset_experimental_formation_energy.csv")
        torch.save(data_list, output_path)
        print(f"Created {len(data_list)} graphs for {dataset_type} set")
        
        return data_list

    def save_lookups(self, output_dir: str) -> None:
        """
        Save the element and precursor lookups to JSON files.
        
        Args:
            output_dir: Directory to save the lookup files
        """
        element_path = os.path.join(output_dir, "element_lookup.json")
        precursor_path = os.path.join(output_dir, "precursor_lookup.json")
        
        with open(element_path, "w") as f:
            json.dump(self.element_lookup.tolist(), f)
        print(f"Saved element lookup to {element_path}")
        
        if self.precursor_lookup.size > 0:
            with open(precursor_path, "w") as f:
                json.dump(self.precursor_lookup.tolist(), f)
            print(f"Saved precursor lookup to {precursor_path}")
        else:
            print("No precursors to save.")

def main():
    """Example usage of the MaterialDatasetProcessor with difficulty selection."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process materials dataset with different difficulty levels')
    parser.add_argument('--difficulty', type=str, choices=['easy', 'medium', 'hard'], 
                       default='medium', help='Difficulty level of the dataset')
    parser.add_argument('--base_path', type=str, default=os.getcwd(),
                       help='Base path for the project')
    args = parser.parse_args()
    
    # Define paths based on difficulty
    data_paths = {
        "easy": {
            "train": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_unfiltered/train.csv",
            "val": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_unfiltered/val.csv",
            "test": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_unfiltered/test.csv"
        },
        "medium": {
            "train": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val_new/train.csv",
            "val": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val_new/val.csv",
            "test": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val_new/test.csv"
        },
        "hard": {
            "train": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val_unique_systems/train.csv", 
            "val": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val_unique_systems/val.csv",
            "test": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val_unique_systems/test.csv"
        }
    }
    
    # Get selected difficulty paths
    print(f"Selected difficulty: {args.difficulty}")
    selected_paths = data_paths[args.difficulty]
    
    # Initialize processor
    matscholar_path = os.path.join(args.base_path, "dataset/matscholar.csv")
    processor = MaterialDatasetProcessor(matscholar_path)
    
    # Prepare lookups using selected dataset
    processor.prepare_dataset_lookups(
        train_path=os.path.join(args.base_path, selected_paths["train"]),
        val_path=os.path.join(args.base_path, selected_paths["val"]),
        test_path=os.path.join(args.base_path, selected_paths["test"])
    )
    
    # Define output directory
    output_dir = os.path.join(f"dataset/our_mpc/{args.difficulty}/")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save lookups
    processor.save_lookups(output_dir)
    
    print(f"\nProcessing {args.difficulty} difficulty dataset:")
    print(f"Number of elements: {processor.element_lookup.shape[0]}")
    print(f"Number of precursors: {processor.precursor_lookup.shape[0]}")
    
    # Create datasets
    for split, path in selected_paths.items():
        full_path = os.path.join(args.base_path, path)
        output_path = os.path.join(output_dir, f"mit_impact_dataset_{args.difficulty}_{split}.pt")
        processor.create_graph_dataset(full_path, split, output_dir)
    
    # Print example data
    train_data = torch.load(os.path.join(output_dir, "mit_impact_dataset_train.pt"))
    print(f"\nTrain data length: {len(train_data)}")
    
    # Print sample with multiple precursors
    for data in train_data:
        if data.y_string_label == "Zr0.3Ti0.7Pb1O3":
            print("\nExample data for Zr0.3Ti0.7Pb1O3:")
            print(f"Number of precursor sets: {data.y_multiple_len}")
            print("\nPrecursor sets:")
            for i in range(data.y_multiple_len.item()):
                precursor_indices = data.y_multiple[i].nonzero().squeeze()
                precursors = [processor.precursor_lookup[idx] for idx in precursor_indices]
                print(f"Set {i + 1}: {precursors}")
            break

if __name__ == "__main__":
    main()

# use like this
# python create_graphs.py --difficulty easy 
# python create_graphs.py --difficulty medium 
# python create_graphs.py --difficulty hard 