import argparse
import concurrent.futures
import torch
import torch.nn as nn
import random
import numpy as np
import os
import sys
from timeit import default_timer as timer
import time as local_time
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import scale
import json
from sklearn.model_selection import train_test_split
from models import Retrieval_Retro
import utils_main
from utils_main import recall_multilabel_multiple, top_k_acc_multiple

torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = "8"

from torch_geometric.data import Batch

def custom_collate_fn(batch):
    # Batch main graphs
    main_graphs = [item[0] for item in batch]
    batched_main_graphs = Batch.from_data_list(main_graphs)
    
    # Handle the first set of additional graphs
    first_additional_graphs = [graph for item in batch for graph in item[1]]
    batched_first_additional_graphs = Batch.from_data_list(first_additional_graphs)

    # Handle the second set of additional graphs
    second_additional_graphs = [graph for item in batch for graph in item[2]]
    batched_second_additional_graphs = Batch.from_data_list(second_additional_graphs)
    
    # Return batched main graphs and both sets of batched additional graphs
    return batched_main_graphs, batched_first_additional_graphs, batched_second_additional_graphs

def seed_everything(seed):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = utils_main.parse_args()
    train_config = utils_main.training_config(args)
    configuration = utils_main.exp_get_name_RetroPLEX(train_config)
    print(f'configuration: {configuration}')


    # GPU setting
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)
    seed_everything(seed=args.seed)

    args.retrieval = 'ours'
    args.split = 'year'

    # Load datasets sequentially
    train_dataset = torch.load(f'./dataset/our/{args.difficulty}/year_train_final_mpc_nre_K_3.pt', map_location=device)
    valid_dataset = torch.load(f'./dataset/our/{args.difficulty}/year_valid_final_mpc_nre_K_3.pt', map_location=device)
    test_dataset = torch.load(f'./dataset/our/{args.difficulty}/year_test_final_mpc_nre_K_3.pt', map_location=device)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, collate_fn = custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size = 1, collate_fn = custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = 1, collate_fn = custom_collate_fn)

    with open(f'./dataset/our_mpc/{args.difficulty}/precursor_lookup.json', 'r') as f:
        precursor_lookup = json.load(f)

    print("Dataset Loaded!")

    gnn = args.gnn
    layers = args.layers
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    n_bond_feat = train_dataset[0][0].edge_attr.shape[1]
    output_dim = train_dataset[0][0].y_multiple.shape[1] 
    embedder = args.embedder
    num_heads = args.num_heads
    t_layers = args.t_layers
    t_layers_sa = args.t_layers_sa
    thres = 'normal'

    f = open(f"./experiments/Retrieval_Retro_{args.difficulty}_{args.batch_size}_{args.lr}_{args.seed}_result.txt", "a")

    if embedder == 'Retrieval_Retro': 
        model = Retrieval_Retro(gnn, layers, input_dim, output_dim, hidden_dim, n_bond_feat, device, t_layers, t_layers_sa, num_heads).to(device)
    else:
        print("############### Wrong Model Name ################")

    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.005)

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded model and optimizer checkpoint from {args.checkpoint_path}")
        print(f"Resuming from epoch {start_epoch}")
        print(f"Top-1 ACC: {checkpoint['top1_acc']:.4f}")
        print(f"Top-3 ACC: {checkpoint['top3_acc']:.4f}")
        print(f"Top-5 ACC: {checkpoint['top5_acc']:.4f}")
        print(f"Top-10 ACC: {checkpoint['top10_acc']:.4f}")
        print(f"Micro Recall: {checkpoint['micro_recall']:.4f}")
        print(f"Macro Recall: {checkpoint['macro_recall']:.4f}")
    else:
        print("No checkpoint path provided")
        start_epoch = 0

    bce_loss = nn.BCELoss()

    train_loss = 0
    num_batch = int(len(train_dataset)/args.batch_size)
    best_acc = 0
    best_epoch = 0
    best_save_epoch = 0
    best_save_acc = 0
    test_macro = 0
    test_micro = 0
    best_acc_list = []
    results_list_of_dics = []
    checkpoint = False
    for epoch in range(start_epoch, args.epochs):
        train_loss = 0
        model.train()
        for bc, batch in enumerate(train_loader):

            y = batch[0].y_lb_one.reshape(len(batch[0].ptr)-1, -1)
            template_output = model(batch)
            loss_template = bce_loss(template_output, y)

            loss = loss_template
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

            sys.stdout.write(f'\r[ epoch {epoch+1}/{args.epochs} | batch {bc}/{num_batch} ] Total Loss : {(train_loss/(bc+1)):.4f}')
            sys.stdout.flush()


        if (epoch + 1) % args.eval == 0 :
            model.eval()
            multi_val_top_1_list = []
            multi_val_top_3_list = []
            multi_val_top_5_list = []
            multi_val_top_10_list = []

            val_micro_rec_list = []
            val_macro_rec_list = []

            with torch.no_grad():

                for bc, batch in enumerate(valid_loader):

                    template_output = model(batch)
                    assert batch[0].y_multiple_len.sum().item() == batch[0].y_multiple.size(0)

                    absolute_indices = torch.cat([torch.tensor([0]).to(device), torch.cumsum(batch[0].y_multiple_len, dim=0)])
                    split_tensors = [batch[0].y_multiple[start:end] for start, end in zip(absolute_indices[:-1], absolute_indices[1:])]

                    multi_label = batch[0].y_multiple

                    # Top-K Accuracy
                    multi_val_top_1_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 1)
                    multi_val_top_1_list.append(multi_val_top_1_scores)
                    multi_val_top_3_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 3)
                    multi_val_top_3_list.append(multi_val_top_3_scores)
                    multi_val_top_5_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 5)
                    multi_val_top_5_list.append(multi_val_top_5_scores)
                    multi_val_top_10_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 10)
                    multi_val_top_10_list.append(multi_val_top_10_scores)

                    # Macro Recall/Micro Recall
                    val_micro_rec, val_macro_rec = recall_multilabel_multiple(split_tensors, template_output.detach().cpu().numpy(), threshold= thres)
                    val_macro_rec_list.append(val_macro_rec)
                    val_micro_rec_list.append(val_micro_rec)

                multi_val_top_1_acc = np.mean(np.concatenate(multi_val_top_1_list))
                multi_val_top_3_acc = np.mean(np.concatenate(multi_val_top_3_list))
                multi_val_top_5_acc = np.mean(np.concatenate(multi_val_top_5_list))
                multi_val_top_10_acc = np.mean(np.concatenate(multi_val_top_10_list))

                val_micro = np.mean(np.concatenate(val_micro_rec_list))
                val_macro = np.mean(np.concatenate(val_macro_rec_list))

                print(f'\n Valid_multi | Epoch: {epoch+1} | Top-1 ACC: {multi_val_top_1_acc:.4f} | Top-3 ACC: {multi_val_top_3_acc:.4f} | Top-5 ACC: {multi_val_top_5_acc:.4f} | Top-10 ACC: {multi_val_top_10_acc:.4f} ')
                print(f'\n Valid Recall | Epoch: {epoch+1} | Micro_Recall: {val_micro:.4f} | Macro_Recall: {val_macro:.4f} ')


                if multi_val_top_5_acc > best_acc:
                    
                    if (epoch + 1) % args.save_interval == 0 and multi_val_top_5_acc > best_save_acc:
                        best_save_acc = multi_val_top_5_acc
                        best_save_epoch = epoch + 1
                        checkpoint = True
                        results_list_of_dics = []


                    best_acc = multi_val_top_5_acc
                    best_epoch = epoch + 1

                    model.eval()

                    multi_top_1_list = []
                    multi_top_3_list = []
                    multi_top_5_list = []
                    multi_top_10_list = []

                    test_micro_rec_list = []
                    test_macro_rec_list = []

                    for bc, batch in enumerate(test_loader):

                        template_output = model(batch)

                        assert batch[0].y_multiple_len.sum().item() == batch[0].y_multiple.size(0)

                        absolute_indices = torch.cat([torch.tensor([0]).to(device), torch.cumsum(batch[0].y_multiple_len, dim=0)])
                        split_tensors = [batch[0].y_multiple[start:end] for start, end in zip(absolute_indices[:-1], absolute_indices[1:])]

                        multi_label = batch[0].y_multiple

                        multi_top_1_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 1)
                        multi_top_1_list.append(multi_top_1_scores)
                        multi_top_3_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 3)
                        multi_top_3_list.append(multi_top_3_scores)
                        multi_top_5_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 5)
                        multi_top_5_list.append(multi_top_5_scores)
                        multi_top_10_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 10)
                        multi_top_10_list.append(multi_top_10_scores)

                        # Macro Recall/Micro Recall
                        test_micro_rec, test_macro_rec = recall_multilabel_multiple(split_tensors, template_output.detach().cpu().numpy(), threshold=thres)
                        test_macro_rec_list.append(test_macro_rec)
                        test_micro_rec_list.append(test_micro_rec)

                        ###############################################################################################################
                        if checkpoint:
                            # Sort probabilities and get corresponding indices for each sample
                            sorted_probs, sorted_indices = torch.sort(template_output, dim=1, descending=True)
                            
                            # Convert to CPU and numpy for easier handling
                            sorted_probs_np = sorted_probs.cpu().numpy()
                            sorted_indices_np = sorted_indices.cpu().numpy()

                            # Get ground truth precursor indices for each batch element
                            gt_ids = batch[0].y_multiple.nonzero(as_tuple=False)[:, 1].cpu().numpy()

                            # Create list of dictionaries for each batch element
                            batch_results = []
                            for i in range(len(batch[0].y_string_label)):
                                result_dict = {
                                    batch[0].y_string_label[i]: {
                                        'gt_precursors': [precursor_lookup[idx] for idx in gt_ids],
                                        'sorted_candidates': [precursor_lookup[idx] for idx in sorted_indices_np[i]],
                                        'sorted_probabilities': sorted_probs_np[i].tolist()
                                    }
                                }
                                batch_results.append(result_dict)
                        
                            results_list_of_dics.extend(batch_results)

                    multi_top_1_acc = np.mean(np.concatenate(multi_top_1_list))
                    multi_top_3_acc = np.mean(np.concatenate(multi_top_3_list))
                    multi_top_5_acc = np.mean(np.concatenate(multi_top_5_list))
                    multi_top_10_acc = np.mean(np.concatenate(multi_top_10_list))

                    test_micro = np.mean(np.concatenate(test_micro_rec_list))
                    test_macro = np.mean(np.concatenate(test_macro_rec_list))

                    print(f'\n Test_multi | Epoch: {epoch+1} | Top-1 ACC: {multi_top_1_acc:.4f} | Top-3 ACC: {multi_top_3_acc:.4f} | Top-5 ACC: {multi_top_5_acc:.4f} | Top-10 ACC: {multi_top_10_acc:.4f} ')
                    print(f'\n Test Recall | Epoch: {epoch+1} | Micro_Recall: {test_micro:.4f} | Macro_Recall: {test_macro:.4f} ')

                    if checkpoint:
                        # Create directories if they don't exist
                        save_dir = f'checkpoints/RR/{args.difficulty}'
                        os.makedirs(save_dir, exist_ok=True)

                        # Fixed checkpoint paths that will be reused
                        checkpoint_path = f'{save_dir}/best_model_{args.seed}.pt'
                        results_path = f'{save_dir}/best_results_{args.seed}.json'

                        # Only save if current accuracy is better than previous best
                        if multi_top_5_acc > best_save_acc:
                            best_save_acc = multi_top_5_acc
                            torch.save({
                                'epoch': epoch + 1,
                                'args': args,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'top1_acc': multi_top_1_acc,
                                'top3_acc': multi_top_3_acc,
                                'top5_acc': multi_top_5_acc,
                                'top10_acc': multi_top_10_acc,
                                'micro_recall': test_micro,
                                'macro_recall': test_macro
                            }, checkpoint_path)
                            with open(results_path, 'w') as f:
                                json.dump(results_list_of_dics, f, indent=4)
                            print(f'New best model saved to {checkpoint_path} (Top-5 ACC: {multi_top_5_acc:.4f})')
                        checkpoint = False

                    best_acc_list.append(multi_top_5_acc)
                    best_state_multi = f'[Best epoch: {best_epoch}] | Top-1 ACC: {multi_top_1_acc:.4f} | Top-3 ACC: {multi_top_3_acc:.4f} | Top-5 ACC: {multi_top_5_acc:.4f} | Top-10 ACC: {multi_top_10_acc:.4f}'
                    best_state_recall = f'[Best epoch: {best_epoch}] | Micro Recall: {test_micro:.4f} | Macro Recall: {test_macro:.4f}'

                    # Early stopping check
                    patience = int(args.es / args.eval)  # Number of evaluations to wait
                    if len(best_acc_list) > patience:
                        # Check if the best accuracy hasn't improved in the last 'patience' evaluations
                        current_best = max(best_acc_list[-patience:])
                        if current_best <= best_acc_list[-patience-1]:
                            print(f'!!Early Stop!! No improvement in last {patience} evaluations')
                            print(f'[FINAL]_MULTI: {best_state_multi}')
                            print(f'[FINAL]_MULTI: {best_state_recall}')
                            f.write("\n")
                            f.write("Early stop!!\n")
                            f.write(configuration)
                            f.write(f"\nbest epoch : {best_epoch}")
                            f.write(f"\nbest Top-1 ACC  MULTI: {multi_top_1_acc:.4f}")
                            f.write(f"\nbest Top-3 ACC MULTI: {multi_top_3_acc:.4f}")
                            f.write(f"\nbest Top-5 ACC MULTI: {multi_top_5_acc:.4f}")
                            f.write(f"\nbest Top-10 ACC MULTI: {multi_top_10_acc:.4f}")
                            f.write(f"\nbest Micro Recall: {test_micro:.4f}")
                            f.write(f"\nbest Macro Recall: {test_macro:.4f}")

                            results_list_of_dics = []

                            for bc, batch in enumerate(test_loader):

                                template_output = model(batch)

                                assert batch[0].y_multiple_len.sum().item() == batch[0].y_multiple.size(0)

                                # Sort probabilities and get corresponding indices for each sample
                                sorted_probs, sorted_indices = torch.sort(template_output, dim=1, descending=True)
                                
                                # Convert to CPU and numpy for easier handling
                                sorted_probs_np = sorted_probs.cpu().numpy()
                                sorted_indices_np = sorted_indices.cpu().numpy()

                                # Get ground truth precursor indices for each batch element
                                gt_ids = batch[0].y_multiple.nonzero(as_tuple=False)[:, 1].cpu().numpy()

                                # Create list of dictionaries for each batch element
                                batch_results = []
                                for i in range(len(batch[0].y_string_label)):
                                    result_dict = {
                                        batch[0].y_string_label[i]: {
                                            'gt_precursors': [precursor_lookup[idx] for idx in gt_ids],
                                            'sorted_candidates': [precursor_lookup[idx] for idx in sorted_indices_np[i]],
                                            'sorted_probabilities': sorted_probs_np[i].tolist()
                                        }
                                    }
                                    batch_results.append(result_dict)
                            
                                results_list_of_dics.extend(batch_results)
                            
                            # Save model at early stopping
                            save_dir = f'checkpoints/RR/{args.difficulty}'
                            os.makedirs(save_dir, exist_ok=True)

                            save_path = f'{save_dir}/early_stopping_epoch{best_epoch}_top5_acc_{multi_top_5_acc:.4f}_{args.seed}.pt'
                            save_path_results = f'{save_dir}/early_stopping_epoch{best_epoch}_results_{args.seed}.json'
                            torch.save({
                                'epoch': best_epoch,
                                'args': args,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'top1_acc': multi_top_1_acc,
                                'top3_acc': multi_top_3_acc, 
                                'top5_acc': multi_top_5_acc,
                                'top10_acc': multi_top_10_acc,
                                'micro_recall': test_micro,
                                'macro_recall': test_macro
                            }, save_path)
                            with open(save_path_results, 'w') as f:
                                json.dump(results_list_of_dics, f, indent=4)
                            print(f'Early stopping: Model and metrics saved to {save_path}')                           
                            sys.exit()


    print(f'Training Done not early stopping')
    print(f'[FINAL]_MULTI: {best_state_multi}')
    f.write("\n")
    f.write("Early stop!!\n")
    f.write(configuration)
    f.write(f"\nbest epoch : {best_epoch}")
    f.write(f"\nbest Top-1 ACC ONE: {multi_top_1_acc:.4f}")
    f.write(f"\nbest Top-3 ACC ONE: {multi_top_3_acc:.4f}")
    f.write(f"\nbest Top-5 ACC ONE: {multi_top_5_acc:.4f}")
    f.write(f"\nbest Top-10 ACC ONE: {multi_top_10_acc:.4f}")
    f.write(f"\nbest Micro Recall: {test_micro:.4f}")
    f.write(f"\nbest Macro Recall: {test_macro:.4f}")
    f.close()

if __name__ == "__main__":

    main()