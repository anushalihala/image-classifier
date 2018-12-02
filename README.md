## Usage

### Train a new network on a data set with train.py

* Basic usage: python train.py data_directory

Prints out training loss, validation loss, and validation accuracy as the network trains

#### Options:

* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory

* Choose architecture: python train.py data_dir --arch "vgg13"

* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20

* Use GPU for training: python train.py data_dir --gpu

### Predict image class with predict.py along with the probability of that class. 

* Basic usage: python predict.py /path/to/image checkpoint

#### Options:

* Return top KK most likely classes: python predict.py input checkpoint --top_k 3

* Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json

* Use GPU for inference: python predict.py input checkpoint --gpu
