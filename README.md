
# Deep Learning for Symbolic Mathematics (reproducibility study)

This is the reproducibility study of the paper [Deep Learning for Symbolic Mathematics](https://arxiv.org/abs/1912.01412) (ICLR 2020). In particular, I tried to reproduce the authors' result on symbolic integration on BWD dataset with greedy decoding. I used PyTorch as the deep learning framework. The video presentation of this project can be found here: [Link](https://www.youtube.com/watch?v=KWJ4feJUxBo). The report of this project is available at [Link](https://drive.google.com/file/d/13Yv66vtI1zl6AOyeDw5eAj20h9OEkaNv/view?usp=sharing).

This repository contains code for:
  - Data handling
  - Training
  - Deep learning model
  - Environment creator
  - Configuration file
  - Logger

I also provide:
- **Map of directories**
    - To run the code without getting an error, add the dataset inside the Dataset/ directory
- **Environment**
    - A conda environment that has the necessary dependencies for this code

## Datasets

The BWD dataset can be found in the original GitHub repository of the paper at: [Link](https://github.com/facebookresearch/SymbolicMathematics)

## Experiments
Here is my experimental setup
- **Hardware**

	| Machine name        | GPU           | RAM  | CPU | Storage |
	| ------------- |:-------------:| -------:| --------:| -------------:|
	| M-K80      | Nvidia K80 | 50 GB | 4 core | 100 GB HDD|
	| M-V100      | Nvidia V100 | 13 GB | 2 core | 100 GB SSD|

- **Experimental Setup**
	| Experiment name        | Epoch size           | Batch size  | Epoch number| Train size | Test size | Validation size |
	| ------------- |:-------------:| -------:| --------:| -------------:| -------------:| -----------------:|
 	| M-K80-4     | 20000 | 32 | 40 | 20000| 500 | 500 |
	| M-V100-3     | 1024 | 32 | 40 | 1024| 32 | 32 |
	| M-V100-4     | 10000 | 32 | 50 | 10000| 32 | 32 |
	| M-V100-5      | 100000 | 256 | 30 | 100000| 5000 | 5000 |
	| M-V100-6    | 100000 | 32 | 92 | 100000 | 500 | 500 |
	| M-V100-7   | 100000 | 532 | 30 | 1000000| 500 | 500 |

## Results
The complete log of each of my experiments can be found in the Results/name_of_the_instance/Dumped/experiment_id/train.log directory. The figures of of the results are available at Figure/ folder. I achieved 95.8% consistent accuracy in M-V100-7 experiment, which was within 2.6% of the reported accuracy. 

## Running the code

If you wish to run my code to verify the experiments, you need to do the following steps after cloning this repository. First, we should download and unzip the BWD dataset from [Link](https://github.com/facebookresearch/SymbolicMathematics). Second, put the extracted files (prim_bwd.train, prim_bwd.test, prim_bwd.valid) in the Dataset/Prime_BWD/ directory. Third, build a conda environment based on the environment.yml file. Fourth, open the terminal in the directory (~/Symbolic-Mathematics/), activate the conda environment with `conda activate DAV`. Now, you can run the code by the following command `python main.py`. This will run an experiment with the default configurations. To alter this configurations, you need to pass the name of the configuration and the new value for that as a command line argument. For example, to run an experiment with the epoch size of 1024 you need to run the following command `python main.py --epoch_size 1000`. To see the list of all configurable variables, you can see the get_parser() function in the main.py file or run the following command  `python main.py --help`.

Note that this environment is the one that I used on my local machine which does not have any GPUs. Thus, I installed the CPU version of PyTorch. If you want to use the GPU version, you need to uninstall this PyTorch version and then install the GPU one. For running on Google Cloud instances, I recommend to use Deep Learning Images available at [Link](https://console.cloud.google.com/marketplace/product/click-to-deploy-images/deeplearning?_ga=2.267303628.391443642.1619619906-1836548465.1616609793&_gac=1.255461882.1616688472.CjwKCAjw6fCCBhBNEiwAem5SOyKMCxW6s1NY5t4CeV3fZEN38KVoinqHaD5ECNDrtcod6sakksLRuRoCz8cQAvD_BwE&pli=1&project=dav-project-308705&folder=&organizationId=) and choose the one with the per-installed PyTorch 1.8 + fast.ai 2.1 (CUDA 11.0). In this case you need to install numexpr and SymPy packages in the base conda environment of that instance via the following commands: ``conda install -c anaconda numexpr`` and `conda install -c anaconda sympy`.  You also need to clone this repo and download the dataset on the instance as well.

By running each experiment, Config class checks to see if there is any `Dumped/` directory available in the `Symbolic-Mathematics` directory. If it does not, it will create one. If the experiment id was not specified, the Config class generates an if and integer number as the id and creates a directory for saving the experiment's outputs in `Symbolic-Mathematics/Dumped/experimend_id`. 

In that directory, model's checkpoint, which is updated after each epoch, the log of the experiment, config instance, and periodic checkpoints will be saved. If you run an existing experiment, the Trainer class first loads the checkpoint.pth and then continues the training from the previous point.  

To make it more clear, the following command will run the M-V100-7 experiment.
`python main.py --epoch_size 100000 --batch_size 32 --max_epoch 30 --train_reload_size 1000000 --test_reload_size 500`

As mentioned in the paper, I evaluate the performance of the model on the whole validation and test set after each epoch. Also, after every 10 epochs, the program stops and asks if you want to continue the training or not. If you choose to stop the experiment, it ends the experiment at that point. 
## References

[**Deep Learning for Symbolic Mathematics**](https://arxiv.org/abs/1912.01412) (ICLR 2020) - Guillaume Lample * and François Charton *

```
@article{lample2019deep,
  title={Deep learning for symbolic mathematics},
  author={Lample, Guillaume and Charton, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:1912.01412},
  year={2019}
}
```
