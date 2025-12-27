# GMAT
## Environment Setup

### Environment Configuration
1. **Creating the Environment**: N

        conda create -n GMAT python=3.8

2. **Activating the Environment**: 

        conda activate GMAT

3. **PyTorch Installation**: Install the required version of PyTorch along with torchvision and torchaudio by running:

        conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch-nightly


## Experiment Setup

- **Configuration**: Specify the base directory and paths for training and evaluation datasets within the `data/datasets.yaml` file.
- **Experiment Configuration**: Use a distinct `.yaml` file for each experiment, located in the `configs` folder. These configuration files encapsulate default parameters aligned with those used in the featured research. Modify these `.yaml` files as necessary to accommodate custom datasets.
- **Guidance on Experiments**: To train GMAT models for classification on PASTIS flod-1, use the following commands for different configurations:
- 
        python train_and_eval/segmentation_training_transf.py --config configs/PASTIS24/GMAT.yaml --device 0
