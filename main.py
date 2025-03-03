#import argparseg
from Config import set_config
from Utils import *
from Experiments import *
from Config import *
from Dataset import *
from Experiments import *
from Metric import *
from Model import *
from Utils import *
from FID import *
from Train import *

#import gc



if __name__ == '__main__':
    class set_args:
        def __init__(self):
            self.seed = 5
            self.state = 'train'
            self.epochs = 10 # training epochs
            self.batch_size = 80
            self.dataset = 'MNIST'#'MNIST'#'CIFAR10' 
            self.split_class= [1]

            #Load different model if TRUE
            self.load_different_model = False
            self.load_training_checkpoint = 30 #30,#200
            self.load_rd_training_checkpoint = 29

            self.sample_dir = 'Result'
            self.checkpoint_dir = 'Result'
            self.M = [1]
            self.I = [0]

            #Reduce training sample
            self.rd = 0.05

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

    df_anchor = expriment1.df_anchor

    print("Experiment 1 Done ......")

    exp2 = Experiment_2(args, model_config, df_anchor)
    train_mean_anchor = exp2.anchor_train_mean_df
    exp2.plot_sim()

    print("experiment 2 Done ......")
    
    # FID

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    InceptionV3_model = InceptionV3([block_idx])
    InceptionV3_model=InceptionV3_model.cuda()

    # create indices
    reduced_sort_idx = create_reduce_idx(args, train_mean_anchor)
    train_new(args, reduced_sort_idx, model_config)

    # load reduced model
    rd_model, rd_sampler, rd_trainer = model_init(args, model_config, rd=True)
    fake = generate_sameple_img(args, model_config, rd_model, rd_sampler)

    fid_dt_train = CUSTOM_DATASET(args, split=True)
    fid_train_dataset, _ = fid_dt_train.load_dataset(custom_trasform=True)

    dt_loader = DataLoader(fid_train_dataset, batch_size=fake.shape[0], shuffle=False)

    for dt, i in dt_loader:
        print(dt.shape)
        break


    # calculate FID
    real = dt
    fretchet_dist = calculate_fretchet(real,fake,InceptionV3_model) 

    print(fretchet_dist)
    grid = make_grid(fake, nrow=8).permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
