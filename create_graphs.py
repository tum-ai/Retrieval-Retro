import os 
import re
import ast
import pandas as pd
import numpy as np
import torch 
from torch_geometric.data import Data
from pymatgen.core import Composition
from typing import Dict, List, Set, Tuple, Optional

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
    
    def prepare_dataset_lookups(self, train_path: str, val_path: Optional[str] = None, 
                              test_path: Optional[str] = None) -> None:
        """
        Create consistent lookups from all datasets.
        
        Args:
            train_path: Path to training data
            val_path: Optional path to validation data
            test_path: Optional path to test data
        """
        # Get unique elements across all datasets
        all_elements = set()
        all_precursors = set()
        
        for path in [p for p in [train_path, val_path, test_path] if p]:
            elements, precursors = self._extract_unique_elements_and_precursors(path)
            all_elements.update(elements)
            all_precursors.update(precursors)
        
        # Create sorted lookups for consistency
        self.element_lookup = np.array(sorted(all_elements))
        self.precursor_lookup = np.array(sorted(all_precursors))
        
        print(f"Found {len(self.element_lookup)} unique elements")
        print(f"Found {len(self.precursor_lookup)} unique precursors")
    
    def _extract_unique_elements_and_precursors(self, csv_path: str) -> Tuple[Set[str], Set[str]]:
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
        
        # Process precursors
        unique_precursors = set()
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
                           output_dir: str) -> List[Data]:
        """
        Create graph dataset from input CSV file.
        
        Args:
            csv_path: Path to input CSV file
            dataset_type: Type of dataset ('train', 'val', or 'test')
            output_dir: Directory to save the processed dataset
        """
        if self.element_lookup is None or self.precursor_lookup is None:
            raise ValueError("Must call prepare_dataset_lookups before creating graphs")
        
        df = pd.read_csv(csv_path)
        
        # Group by target_formula to get all precursor sets for each target
        grouped_df = df.groupby('target_formula')['precursor_formulas'].agg(list).reset_index()
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
                        y_multiple_len=torch.tensor([num_precursor_sets]),
                        y_lb_freq=y_lb_all,
                        y_lb_avg=y_lb_all,
                        y_lb_all=y_lb_all,
                        y_lb_one=y_multiple[i],  # Each combination becomes y_lb_one for a separate graph
                        y_string_label=row['target_formula']
                    )
                    data_list_for_target.append(data)
                
                data_list.extend(data_list_for_target)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        # Save dataset
        output_path = os.path.join(output_dir, f"mit_impact_dataset_{dataset_type}.pt")
        torch.save(data_list, output_path)
        print(f"Created {len(data_list)} graphs for {dataset_type} set")
        
        return data_list

def main():
    """Example usage of the MaterialDatasetProcessor."""
    base_path = os.getcwd()
    matscholar_path = os.path.join(base_path, "dataset/matscholar.csv")
    
    # Initialize processor
    processor = MaterialDatasetProcessor(matscholar_path)
    
    # Define dataset paths
    data_paths = {
        "train": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val/train_data_up_to_2014.csv",
        "val": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val/val_data_up_to_2014.csv",
        "test": "/home/thorben/code/mit/PrecursorRanker/data/dataset/dataset_w_candidates_w_val/test_data_after_2014.csv"
    }
    
    # Prepare lookups using all datasets
    processor.prepare_dataset_lookups(
        train_path=data_paths["train"],
        val_path=data_paths["val"],
        test_path=data_paths["test"]
    )

    print(processor.element_lookup)
    print(processor.element_lookup.shape)
    print(processor.precursor_lookup)
    print(processor.precursor_lookup.shape)
    
    # Create datasets
    output_dir = os.path.join(base_path, "dataset")
    
    for split, path in data_paths.items():
        processor.create_graph_dataset(path, split, output_dir)

    # print train data
    train_data = torch.load(os.path.join(output_dir, "mit_impact_dataset_train.pt"))
    # print train data length
    print(f"Train data length: {len(train_data)}")
    #print(train_data[0])
    #print sample with multiple precursors (y_multiple_len > 1)
    # print for target formula: "Zr0.3Ti0.7Pb1O3" then print y_multiple index and value as well as the precursor for this index
    for data in train_data:
        if data.y_string_label == "Zr0.3Ti0.7Pb1O3":
            print(data)
            print(f"Number of precursor sets (y_multiple_len): {data.y_multiple_len}")
            print("\nPrecursor sets (y_multiple):")
            for i in range(data.y_multiple_len.item()):
                precursor_indices = data.y_multiple[i].nonzero().squeeze()
                precursors = [processor.precursor_lookup[idx] for idx in precursor_indices]
                print(f"Set {i + 1}: {precursors}")

if __name__ == "__main__":
    main()
