import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import os
import sys
import time as local_time
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from models import MPC, MultiLossLayer, CircleLoss
import utils_mpc
from utils_main import recall_multilabel_multiple, top_k_acc_multiple
from tensorboardX import SummaryWriter
from torch_geometric.data import Batch

torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = "4"

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
    args = utils_mpc.parse_args()
    train_config = utils_mpc.training_config(args)
    configuration = utils_mpc.train_mpc(train_config)
    print(f'configuration: {configuration}')

    # GPU setting
    args.embedder = 'mpc'

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print(device)
    seed_everything(seed=args.seed)

    train_dataset = torch.load('./dataset/year/year_train_mpc.pt',map_location=device)
    valid_dataset = torch.load('./dataset/year/year_valid_mpc.pt',map_location=device)
    test_dataset = torch.load('./dataset/year/year_test_mpc.pt',map_location=device)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True) 
    valid_loader = DataLoader(valid_dataset, batch_size = 1)
    test_loader = DataLoader(test_dataset, batch_size = 1)

    print("Dataset Loaded!")


    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    output_dim = train_dataset[0].y_multiple.shape[1] 
    embedder = args.embedder
    loss_type = args.loss
    loss_type = 'adaptive'
    task_names = ['multi-label', 'reconstruction']

    f = open(f"./experiments/mpc_{args.loss}_{args.split}_{args.embedder}_result.txt", "a")
    
    if embedder == 'mpc':
        model = MPC(input_dim, hidden_dim, output_dim, device).to(device)
    else:
        print("############### Wrong Model Name ################")

    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss()
    circle_loss = CircleLoss()
    adaptive_loss = MultiLossLayer(task_names, device)


    bce_loss = nn.BCELoss()

    train_loss = 0
    num_batch = int(len(train_dataset)/args.batch_size)
    best_acc = 0
    best_epoch = 0
    test_macro = 0
    test_micro = 0
    best_acc_list = []
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()

        for bc, batch in enumerate(train_loader):


            batch.to(device)

            multi_label, emb, reconstruction = model(batch, None)

            y = batch.y_lb_one.reshape(len(batch.ptr)-1, -1)
            if loss_type == "bce":
                loss_template = bce_loss(multi_label, y)
                total_loss = loss_template

            else:
                y_recon = batch.comp_fea.reshape(len(batch.ptr)-1, -1)

                mpc_loss = circle_loss(y, multi_label)
                reconstruction_loss = mse_loss(reconstruction, y_recon) 
                loss_sum = torch.cat([mpc_loss.unsqueeze(0), reconstruction_loss.unsqueeze(0)], dim=-1) # or cat?
                total_loss = adaptive_loss(loss_sum)

            loss = total_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

            sys.stdout.write(f'\r[ epoch {epoch+1}/{args.epochs} | batch {bc+1}/{num_batch} ] Total Loss : {(train_loss/(bc+1)):.4f}')
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

                    batch.to(device)

                    template_output, emb, reconstruction = model(batch, None)
                    assert batch.y_multiple_len.sum().item() == batch.y_multiple.size(0)

                    absolute_indices = torch.cat([torch.tensor([0]).to(device), torch.cumsum(batch.y_multiple_len, dim=0)])
                    split_tensors = [batch.y_multiple[start:end] for start, end in zip(absolute_indices[:-1], absolute_indices[1:])]

                    multi_label = batch.y_multiple

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
                    val_micro_rec, val_macro_rec = recall_multilabel_multiple(split_tensors, template_output.detach().cpu().numpy())
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
                    best_acc = multi_val_top_5_acc
                    best_epoch = epoch + 1

                    model.eval()

                    multi_top_1_list = []
                    multi_top_3_list = []
                    multi_top_5_list = []
                    multi_top_10_list = []

                    test_micro_rec_list = []
                    test_macro_rec_list = []


                    with torch.no_grad():
                        for bc, batch in enumerate(test_loader):
                            batch.to(device)

                            template_output, emb, reconstruction = model(batch, None)
                            assert batch.y_multiple_len.sum().item() == batch.y_multiple.size(0)

                            absolute_indices = torch.cat([torch.tensor([0]).to(device), torch.cumsum(batch.y_multiple_len, dim=0)])
                            split_tensors = [batch.y_multiple[start:end] for start, end in zip(absolute_indices[:-1], absolute_indices[1:])]

                            multi_label = batch.y_multiple

                            multi_top_1_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 1)
                            multi_top_1_list.append(multi_top_1_scores)
                            multi_top_3_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 3)
                            multi_top_3_list.append(multi_top_3_scores)
                            multi_top_5_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 5)
                            multi_top_5_list.append(multi_top_5_scores)
                            multi_top_10_scores = top_k_acc_multiple(multi_label, template_output.detach().cpu().numpy(), 10)
                            multi_top_10_list.append(multi_top_10_scores)

                            # Macro Recall/Micro Recall
                            test_micro_rec, test_macro_rec = recall_multilabel_multiple(split_tensors, template_output.detach().cpu().numpy())
                            test_macro_rec_list.append(test_macro_rec)
                            test_micro_rec_list.append(test_micro_rec)

                        multi_top_1_acc = np.mean(np.concatenate(multi_top_1_list))
                        multi_top_3_acc = np.mean(np.concatenate(multi_top_3_list))
                        multi_top_5_acc = np.mean(np.concatenate(multi_top_5_list))
                        multi_top_10_acc = np.mean(np.concatenate(multi_top_10_list))

                        test_micro = np.mean(np.concatenate(test_micro_rec_list))
                        test_macro = np.mean(np.concatenate(test_macro_rec_list))

                        print(f'\n Test_multi | Epoch: {epoch+1} | Top-1 ACC: {multi_top_1_acc:.4f} | Top-3 ACC: {multi_top_3_acc:.4f} | Top-5 ACC: {multi_top_5_acc:.4f} | Top-10 ACC: {multi_top_10_acc:.4f} ')
                        print(f'\n Test Recall | Epoch: {epoch+1} | Micro_Recall: {test_micro:.4f} | Macro_Recall: {test_macro:.4f} ')

                try:
                    if multi_top_5_acc:
                        best_acc_list.append(multi_top_5_acc)
                        best_state_multi = f'[Best epoch: {best_epoch}] | Top-1 ACC: {multi_top_1_acc:.4f} | Top-3 ACC: {multi_top_3_acc:.4f} | Top-5 ACC: {multi_top_5_acc:.4f} | Top-10 ACC: {multi_top_10_acc:.4f}'
                        best_state_recall = f'[Best epoch: {best_epoch}] | Micro Recall: {test_micro:.4f} | Macro Recall: {test_macro:.4f}'
                except:
                    continue

                if len(best_acc_list) > int(args.es / args.eval):
                    if best_acc_list[-1] == best_acc_list[-int(args.es / args.eval)]:
                        print(f'!!Early Stop!!')
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
