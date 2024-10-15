# Retrieval-Retro: Retrieval-based Inorganic Retrosynthesis with Expert Knowledge
<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://nips.cc/" alt="Main conference">
        <img src="https://img.shields.io/badge/2024 Neurips-blue" /></a>
</p>
The offical source code for Retrieval-Retro: Retrieval-based Inorganic Retrosynthesis with Expert Knowledge, accepted at 2024 NeurIPS main conference.
 
 
### Overview
While inorganic retrosynthesis planning is essential in the field of chemical science, the application of machine learning in this area has been notably less explored compared to organic retrosynthesis planning. In this paper, we propose Retrieval-Retro for inorganic retrosynthesis planning, which implicitly extracts the precursor information of reference materials that are retrieved from the knowledge base regarding domain expertise in the field. Specifically, instead of directly employing the precursor information of reference materials, we propose implicitly extracting it with various attention layers, which enables the model to learn novel synthesis recipes more effectively. Moreover, during retrieval, we consider the thermodynamic relationship between target material and precursors, which is essential domain expertise in identifying the most probable precursor set among various options. Extensive experiments demonstrate the superiority of Retrieval-Retro in
retrosynthesis planning, especially in discovering novel synthesis recipes, which is crucial for materials discovery.

### Training retrievers part of Retrieval-Retro 


### Overall Architecture of Retrieval-Retro 



#### Dataset
You can dowload year split dataset in this drive [link](https://drive.google.com/drive/folders/1hDCVjT50au_f5fVeZxF_4U2nF6GmhO74?usp=sharing)  
After download dataset, you can put datasets into dataset folder  
(We upload our dataset into drive link due to file size)

#### Run model
Run `main_Retrieval_Retro.py` for training our Retrieval-Retro (after download the dataset)

### Models
#### Retrieval-Retro
`Retrieval_Retro.py`: Our proposed model: Retrieval-Retro

### Others
#### MPC
`MPC.py`: MPC Retriever
`train_mpc.py`: For training MPC Retriever
`retrieval_mpc.py`: For calculating the cosine similarity and save top-k retrieved materials from MPC retriever
`utils_mpc.py`: utils for MPC Retriever training

#### NRE
`GraphNetwork.py`: Formation Energy predictor GraphNetwork & GraphNetwork backbone
`pretrain_nre.py`: For training NRE Retriever (Formation Energy predictor) 
`calculate_gibbs.py`: For calculating the Gibbs free energy between the target material and precursor set of materials in the knowledge base
`retrieval_nre.py`: Save top-k retrieved materials from NRE retriever
`utils_nre.py`: utils for NRE Retriever (Formation Energy predictor) training

You can use `collate.py` to collate retrieved data from MPC & NRE retrievers for constructing final dataset

### Hyperparameters  

`--layers:` Number of GNN layers in the Retrieval-Retro

`--t_layers:` Number of cross-attention layers in the Retrieval-Retro

`--t_layers:` Number of self-attention layers in the Retrieval-Retro

`--embedder:` Selecting embedder   

`--hidden:` Size of hidden dimension

`--epochs:`  Number of epochs for training the model

`--lr:` Learning rate for training the model  

`--es:` Early stopping criteria  

`--eval:` Evaluation step  

`--K:` Number of retrieved materials
