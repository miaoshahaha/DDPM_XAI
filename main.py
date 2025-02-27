#import argparseg
from Config import set_config
from Utils import *
from Experiments import *


if __name__ == '__main__':
    class set_args:
        def __init__(self):
            self.seed = 5
            self.state = 'train'
            self.epochs = 0
            self.batch_size = 80
            self.dataset = 'MNIST'#'MNIST'#'CIFAR10' 
            self.split_class= [1]
            self.load_training_checkpoint = 30 #30,#200
            self.sample_dir = 'Result'
            self.checkpoint_dir = 'Result'
            self.M = [1]
            self.I = [0]

            self.save_weight_dir = 'Result/CheckpointsMNIST/'

            
    
    directory_init()

    args = set_args()
    seed_everything(args.seed) 

    """Merge both these"""
    model_config = set_config(args, custom_dataset=args.dataset) 
    
    model, sampler, trainer = model_init(args, model_config)

    """Handle these afterward"""
    #h_info_sampledImgs = sampler.load_h_information()
    #xt_info_sampledImgs = sampler.load_xt_information()

    expriment1 = Experiment_1(args, model_config, model, sampler, trainer)
    expriment1.plot_sim()
