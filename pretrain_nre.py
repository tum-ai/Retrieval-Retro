import torch
import torch.nn as nn
import numpy as np
import random
from torch_geometric.loader import DataLoader
import utils_nre
from utils_nre import r2
import os
import sys
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import time as local_time
from torch_scatter import scatter_mean, scatter_sum

from models import GraphNetwork_prop


# limit CPU usage
torch.set_num_threads(4)

def test_var(model, pretrain, data_loader,l1, r2, device):
    model.eval()

    with torch.no_grad():
        loss_rmse, loss_mse, loss_mae,  = 0, 0, 0
        for bc, batch in enumerate(data_loader):

            batch.to(device)
            preds, emb = model(batch)

            if pretrain == 'formation_ft' or pretrain == 'formation':
                y = batch.form_e.reshape(len(batch.form_e), -1)

            elif pretrain == 'formation_exp':
                y = batch.y_exp_form.reshape(len(batch.y_exp_form), -1) 

            mse = ((y - preds)**2).mean(dim = 1)
            rmse = torch.sqrt(mse)
    
            loss_mse += mse.mean()
            loss_rmse += rmse.mean()
            
            mae = l1(preds, y).cpu()
            loss_mae += mae

    return loss_rmse/(bc + 1), loss_mse/(bc+1), loss_mae/(bc+1)



def main():
    
    args = utils_nre.parse_args()
    train_config = utils_nre.training_config(args)

    # Seed Setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    configuration = utils_nre.train_predictor(train_config)
    print("{}".format(configuration))

    # GPU setting
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) 
    print(device)
    from sklearn.model_selection import train_test_split

    # Load dataset
    if args.pretrain == "formation_exp":
        dataset = torch.load('/home/thorben/code/mit/Retrieval-Retro/dataset/nre/mit_impact_dataset_experimental_formation_energy.pt', map_location=device)
    elif args.pretrain == 'formation_ft' or args.pretrain == 'formation':
        dataset = torch.load('./dataset/materials_project_formation_energy.pt', map_location=device)
    else:
        print("wrong pretraining dataset")

    train_ratio = 0.80
    validation_ratio = 0.10
    test_ratio = 0.10

    train_dataset, test_dataset = train_test_split(dataset, test_size=1 - train_ratio, random_state=args.seed)
    valid_dataset, test_dataset = train_test_split(test_dataset, test_size=test_ratio/(test_ratio + validation_ratio), random_state=args.seed)
    print(f'train_dataset_len:{len(train_dataset)}')
    print(f'valid_dataset_len:{len(valid_dataset)}')
    print(f'test_dataset_len:{len(test_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = 1)
    test_loader = DataLoader(test_dataset, batch_size = 1)
    print("Dataset Loaded!")

    embedder = args.embedder.lower()
    n_hidden = args.hidden
    n_atom_feat = train_dataset[0].x.shape[1]
    n_bond_feat = train_dataset[0].edge_attr.shape[1]


    # Model selection

    if embedder == "graphnetwork":
        model = GraphNetwork_prop(args.layers, n_atom_feat, n_bond_feat, n_hidden, device).to(device)
    else :
        print("error occured : Inappropriate model name")
    print(model)

    f = open(f"./experiments/predictor_{args.pretrain}.txt", "a")
    
    ########################## You can use the checkpoint of pretrained model for transfer learning###########################
    # For the dataset size, we can't upload full dft-calculated data
    checkpoint = torch.load("/home/thorben/code/mit/Retrieval-Retro/checkpoints/nre/nre_pretrain_only_on_materials_project_formation_energy/model_best_mae_0.1883.pt", map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f'\nModel Weight Loaded')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion_2 = nn.L1Loss()

    train_loss = 0
    best_mae = 1000
    num_batch = int(len(train_dataset)/args.batch_size)

    best_losses = list()
    for epoch in range(args.epochs):

        train_loss = 0
        start = timer()
        model.train()

        for bc, batch in enumerate(train_loader):        
            batch.to(device)
            preds, emb = model(batch) 

            if args.pretrain == 'formation_ft' or args.pretrain == 'formation':
                y = batch.form_e.reshape(len(batch.form_e), -1)

            elif args.pretrain == 'formation_exp':
                y = batch.y_exp_form.reshape(len(batch.y_exp_form), -1)

            loss = criterion_2(preds, y).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

            sys.stdout.write('\r[ epoch {}/{} | batch {}/{} ] Total Loss : {:.4f} '.format(epoch + 1, args.epochs, bc + 1, num_batch + 1, (train_loss/(bc+1))))
            sys.stdout.flush()

        # Add periodic checkpointing
        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            save_dir = './checkpoints/mit_nre_finetune_experimental_formation_energy/'
            os.makedirs(save_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss/(bc+1),
                'args': args
            }
            torch.save(checkpoint, os.path.join(save_dir, f'model_epoch_{epoch+1}.pt'))
            print(f"\nSaved checkpoint at epoch {epoch+1}")

        if (epoch + 1) % args.eval == 0 :
            
            time = (timer() - start)
            print("\ntraining time per epoch : {:.4f} sec".format(time))
            
            #valid
            valid_rmse, valid_mse,valid_mae = test_var(model, args.pretrain, valid_loader, criterion_2, r2, device)
            print("\n[ {} epochs ]valid_rmse:{:.4f}|valid_mse:{:.4f}|valid_mae:{:.4f}".format(epoch + 1, valid_rmse, valid_mse,valid_mae))

            if  valid_mae < best_mae:
                best_mae = valid_mae
                best_epoch = epoch + 1 
                test_rmse, test_mse,test_mae = test_var(model, args.pretrain, test_loader, criterion_2, r2, device)
                print("\n[ {} epochs ]test_rmse:{:.4f}|test_mse:{:.4f}|test_mae:{:.4f}".format(epoch + 1, test_rmse, test_mse,test_mae))

            best_losses.append(best_mae)
            st_best = '**[Best epoch: {}] Best RMSE: {:.4f}|Best MSE: {:.4f} |Best MAE: {:.4f}**\n'.format(best_epoch, test_rmse,test_mse, test_mae)
            print(st_best)
            if len(best_losses) > int(args.es / args.eval):
                if best_losses[-1] == best_losses[-int(args.es / 5)]:
                    
                    print("Early stop!!")
                    print("[Final] {}".format(st_best))
                    f.write("\n")
                    f.write("Early stop!!\n")
                    f.write(configuration)
                    f.write("\nbest epoch : {} \n".format(best_epoch))
                    f.write("best RMSE : {:.4f} \n".format(test_rmse))
                    f.write("best MSE : {:.4f} \n".format(test_mse))
                    f.write("best MAE : {:.4f} \n".format(test_mae))
                    
                    # Save model checkpoint
                    save_dir = './checkpoints/nre/mit_nre_finetune_experimental_formation_energy/'
                    os.makedirs(save_dir, exist_ok=True)
                    checkpoint = {
                        'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_mae': best_mae,
                        'args': args
                    }
                    torch.save(checkpoint, os.path.join(save_dir, f'model_best_mae_{best_mae:.4f}.pt'))
                    print(f"Saved best model to {save_dir}")
                    
                    sys.exit()
        
    print("\ntraining done!")
    print("[Final] {}".format(st_best))
    # write experimental results
    f.write("\n")
    f.write(configuration)
    f.write("\nbest epoch : {} \n".format(best_epoch))
    f.write("best RMSE : {:.4f} \n".format(test_rmse))
    f.write("best MSE : {:.4f} \n".format(test_mse))
    f.write("best MAE : {:.4f} \n".format(test_mae))
    f.close()


if __name__ == "__main__" :
    main()
