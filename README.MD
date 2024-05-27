# Implicit Precursor Extraction with Expert Retriever for Inorganic Retrosynthesis

The offical source code for Implicit Precursor Extraction with Expert Retriever for Inorganic Retrosynthesis
 
### Overview
While inorganic retrosynthesis planning is essential in the field of chemical science, the application of machine learning in this area has been notably less explored compared to organic retrosynthesis planning. In this paper, we propose RetroPLEX for inorganic Retrosynthesis planning, which imPLicitly EXtracts the precursor information of reference materials that are retrieved from the knowledge base regarding domain expertise in the field. Specifically, instead of directly employing the precursor information of reference materials, we propose implicitly extracting it with various attention layers, which enables the model to learn novel synthesis recipes more effectively. Moreover, during retrieval, we consider the thermodynamic relationship between target material and precursors, which is essential domain expertise in identifying the most probable precursor set among various options. Extensive experiments demonstrate the superiority of RetroPLEX in
retrosynthesis planning, especially in discovering novel synthesis recipes, which is crucial for materials discovery.

### RetroPLEX
#### Dataset
You can dowload year split dataset in this drive [link](https://drive.google.com/drive/folders/1hDCVjT50au_f5fVeZxF_4U2nF6GmhO74?usp=sharing)  
After download dataset, you can put datasets into dataset folder  
(We upload our dataset into drive link due to file size)

#### Run model
Run `main_RetroPLEX.py` for training our RetroPLEX

### Models
#### RetroPLEX
`RetroPLEX.py`: Our proposed model: RetroPLEX


### Others
#### MPC
`MPC.py`: MPC Retriever
`train_mpc.py`: For training MPC Retriever
`utils_mpc.py`: utils for MPC Retriever training

#### GraphNetwork
`GraphNetwork.py`: Formation Energy predictor GraphNetwork & GraphNetwork backbone
`train_mpc.py`: For training NRE Retriever (Formation Energy predictor) training
`utils_nre.py`: utils for NRE Retriever (Formation Energy predictor) training

### Hyperparameters  

`--layers:` Number of GNN layers in RetroPLEX model  

`--t_layers:` Number of cross-attention layers in RetroPLEX model  

`--t_layers:` Number of self-attention layers in RetroPLEX model  

`--embedder:` Selecting embedder   

`--hidden:` Size of hidden dimension

`--epochs:`  Number of epochs for training the model

`--lr:` Learning rate for training the model  

`--es:` Early Stopping Criteria  

`--eval:` Evaluation Step  

`--K:` Number of retrieved materials
