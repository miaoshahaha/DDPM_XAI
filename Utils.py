import random
import os 
import pandas as pd 
import numpy as np
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import torch
import torch.nn as nn 
from torch.nn import functional as F

from Model import UNet, GaussianDiffusionSampler, GaussianDiffusionTrainer
from Dataset import CUSTOM_DATASET
from Config import set_config

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_sample_h_space_and_xt(args, modelConfig):
    """
    Output : 
    -- h_info_sampledImgs : output h_space from the model.
    """
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
        modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
        size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]

        h_info_sampledImgs = sampler.load_h_information()
        xt_info_sampledImgs = sampler.load_xt_information()


    return h_info_sampledImgs, xt_info_sampledImgs

"""
def get_sample_xt(args, modelConfig):
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
        modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
        size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
"""

def save_sample_h_space_and_xt(args, modelConfig, mode='save'):
    """
    Save the h_space information to the pickle file.
    """
    #load_models = []
    if mode == 'save':
        model_h_info = {}
        model_xt = {}
        for _, _, models in os.walk(os.path.join(os.getcwd(),modelConfig["save_weight_dir"])):
            load_models = models
        for model in load_models:
            model_label = model.split('_')[0]
            modelConfig["test_load_weight"] = model
            model_h_info[model_label], model_xt[model_label] = get_sample_h_space_and_xt(args, modelConfig)

        h_info_f_name = 'Result/hInfo'+args.dataset+'/hInfo.pickle'
        xt_info_f_name = 'Result/hInfo'+args.dataset+'/xtInfo.pickle'
        #f_name = 'Result/hInfo'+args.dataset+'/hInfo'+str(args.split_class)+'.pickle'
        with open(h_info_f_name, 'wb') as handle:
            pickle.dump(model_h_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(xt_info_f_name, 'wb') as handle:
            pickle.dump(model_xt, handle, protocol=pickle.HIGHEST_PROTOCOL ) 
        #h_df = pd.DataFrame.from_dict(model_h_info, orient='index')
        #h_df.to_csv('Result/hInfo'+args.dataset+'/hInfo.csv')
    elif mode == 'load_h':
        f_name = 'Result/hInfo'+args.dataset+'/hInfo.pickle'
        #f_name = 'Result/hInfo'+args.dataset+'/hInfo'+str(args.split_class)+'.pickle'
        with open(f_name, 'rb') as handle:
            h_info = pickle.load(handle)
        return h_info
    elif mode == 'load_xt':
        f_name = 'Result/hInfo'+args.dataset+'/xtInfo.pickle'
        #f_name = 'Result/hInfo'+args.dataset+'/hInfo'+str(args.split_class)+'.pickle'
        with open(f_name, 'rb') as handle:
            xt_info = pickle.load(handle)
        return xt_info
        

def get_InterferenceImgs(args, modelConfig, h_info_sampledImgs):
    """
    Get interference images from timesteps 999 down to 99. Total 10 iterations.
    input :
    -- h_info_sampledImgs : run get_sample_h_space() first to collect the data.
    output :
    -- interferenceImgs : Images interfered by other model's h_space.
    """
    interferenceImgs = {}
    device = torch.device(modelConfig["device"])
    noisyImage_1 = torch.randn(
        size=[modelConfig["batch_size"], 3, 32, 32], device=device)
    
    model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
    ckpt = torch.load(os.path.join(
    modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
    model.load_state_dict(ckpt)
    print("model load weight done.")
    model.eval()

    for i in range(0,10):
        with torch.no_grad():
            if i == 0: # Sample from the original noise
                sampler = GaussianDiffusionSampler(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
            else: # Sample from the interference noise
                sampler = GaussianDiffusionSampler(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], h_info_sampledImgs, timestep_edition=i).to(device)
            # Sampled from standard normal distribution
            #noisyImage_1 = torch.randn(
            #size=[modelConfig["batch_size"], 3, 32, 32], device=device)
            sampledImgs_1 = sampler(noisyImage_1)
            sampledImgs_1 = sampledImgs_1 * 0.5 + 0.5  # [0 ~ 1]
            interferenceImgs[i] = sampledImgs_1

        # Grid dimensions
        n_cols = 5  # Number of columns
        n_rows = 2#len(sampledImgs) // n_cols + (len(sampledImgs) % n_cols > 0)  # Calculate number of rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))  # Create subplots

        # Flatten axes array for easy iteration
        axes = axes.flatten()

        t = i
        # Loop through the images and axes
        print(f'T = {t}')
        for i, ax in enumerate(axes):
            if i < len(sampledImgs_1):  # Check if there's an image to show
                #ax.imshow(interferenceImgs[t][i].cpu().numpy()[0], cmap='gray')  # Show the ith image
                ax.imshow(interferenceImgs[t][i].cpu().numpy().transpose(1,2,0))  # Show the ith image
                ax.axis('off')  # Optionally, turn off axis
            else:
                ax.axis('off')  # Turn off unused axes

        plt.subplots_adjust(wspace=0, hspace=-0.74)  # Adjust horizontal and vertical spacing
        plt.savefig(f'Result/InterferenceImg/M{args.M}_I{args.I}_T{1000 - t*100}ha.png', bbox_inches='tight', dpi=300)  # Save with tight bounding box and high resolution

        #plt.tight_layout()  # Adjust layout to prevent overlap
        #plt.show()

    return interferenceImgs

def directory_init():
    """
    Make necessary directories to run the code.
    """
    curr_dir = os.path.dirname(__file__)
    os.makedirs('Result', exist_ok=True)
    
    res_dir = os.path.join(os.path.dirname(__file__), 'Result') 

    #os.makedirs(res_dir + '/InterferenceImg', exist_ok=True)
    os.makedirs(res_dir + '/CheckpointsMNIST', exist_ok=True)
    os.makedirs(res_dir + '/CheckpointsImagenet', exist_ok=True)
    os.makedirs(res_dir + '/CheckpointsCIFAR10', exist_ok=True)
    
    os.makedirs(res_dir + '/ResultsMNIST', exist_ok=True)
    os.makedirs(res_dir + '/ResultsImagenet', exist_ok=True)
    os.makedirs(res_dir + '/ResultsCIFAR10', exist_ok=True)
    
    #os.makedirs(res_dir + '/SampledImgsMNIST', exist_ok=True)
    #os.makedirs(res_dir + '/SampledImgsImagenet', exist_ok=True)
    #os.makedirs(res_dir + '/SampledImgsCIFAR10', exist_ok=True)

    #os.makedirs(res_dir + '/hInfoMNIST', exist_ok=True)
    #os.makedirs(res_dir + '/hInfoImagenet', exist_ok=True)
    #os.makedirs(res_dir + '/hInfoCIFAR10', exist_ok=True)

    
    
    
    #args.checkpoint_dir + args.dataset


"""
Plot image
"""

def plot_h_comparison(args, h_info, base=0, t=5):
    
    output_kld = {}
    output_mse = {}
    #curr_target = dt['[0]'][3] # T = 700
    #curr_target = dt['[0]'][5] # T = 500
    curr_target = h_info[f'[{base}]'][t] # T = 300



    for key, value in h_info.items():
        output_kld[key] = []
        output_mse[key] = []

        for i in range(100):
            rd_1 = random.randint(1, len(curr_target) - 1)
            rd_2 = random.randint(1, len(curr_target) - 1)
            curr_input = h_info[f'{key}'][t] # T = 700

            kl_loss = nn.KLDivLoss(reduction='batchmean')
            kld_input = F.log_softmax(curr_input[rd_1], dim=1)
            kld_target = F.softmax(curr_target[rd_2], dim=1)

            output_kld[key].append(round(kl_loss(kld_input, kld_target).item(), 4))

            mse_input = curr_input[rd_1]
            mse_target = curr_target[rd_2]

            mse_loss = nn.MSELoss()
            output_mse[key].append(round(mse_loss(mse_input, mse_target).item(), 4))
            #print(output_mse)
            #output_mse[key].append(round(kl_loss(kld_input, kld_target).item(), 4))
        
    sns.set_theme(style="darkgrid")

    output_kld_df = pd.DataFrame(output_kld)
    output_mse_df = pd.DataFrame(output_mse)
    sns.boxplot(data=output_kld_df, width=.3)
    plt.ylim(0,1)
    plt.ylabel('kld', fontsize=16)
    plt.title(f'kld - Compare with [{base}] at timestep : {str(1000 - t * 100)}')
    plt.draw()
    plt.savefig(f'Result/hInfo{args.dataset}/kld_h_compare{str(1000 - t*100)}.png')
    plt.pause(1)
    plt.close()

    sns.boxplot(data=output_mse_df, width=.3)
    plt.ylim(0,1)
    plt.ylabel('mse', fontsize=16)
    plt.title(f'mse - Compare with [{base}] at timestep : {str(1000 - t * 100)}')
    plt.draw()
    plt.savefig(f'Result/hInfo{args.dataset}/mse_h_compare{str(1000 - t*100)}.png')
    plt.pause(1)
    plt.close()
    """
    output_df = pd.concat([output_kld_df, output_mse_df],keys = ['kld','mse'],axis=1)
    print(output_df)
    sns.scatterplot(data=output_df, x='mse', y='kld')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('mse', fontsize=16)
    plt.ylabel('kld', fontsize=16)
    plt.title(f'Compare with [{base}] at timestep : {str(1000 - t * 100)}')
    plt.draw()
    plt.savefig(f'Result/hInfo{args.dataset}/all_h_compare{str(1000 - t*100)}.png')
    plt.pause(1)
    plt.close()
    """

class h_info_analysis:
    def __init__(self, model_config, args, h_info, base=0):
        
        #self.model = model
        #self.sampler = sampler
        self.args = args
        self.xt = self._collect_xt(model_config)
        self.model_config = model_config
        self.h_info = h_info
        self.base = base
        self.h_df = self._collect_h_info()
        #self.h_df = pd.DataFrame.from_dict(self.h_dict)
        self.x_boundary = 0
        self.y_boundary = 0
        
        
    def plot_h(self):
        plt.figure(figsize=(16,9))
        h_df = self.h_df
        ax = sns.scatterplot(data=h_df, x='mse', y='kld', hue='target')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title(f'Scatter Plot - Different Classes Compare with {self.args.dataset}{self.args.split_class}', fontsize = 18)
        plt.savefig(f'Result/hInfo{self.args.dataset}/scatter_plot_target_{self.args.dataset}_{self.args.split_class}.png')

        plt.figure(figsize=(16,9))
        h_df = self.h_df
        ax = sns.scatterplot(data=h_df, x='mse', y='kld', hue='quadrant')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title(f'Scatter Plot - Different Quadrant Compare with {self.args.dataset}{self.args.split_class}', fontsize = 18)
        plt.savefig(f'Result/hInfo{self.args.dataset}/scatter_plot_quadrant_{self.args.dataset}_{self.args.split_class}.png')
        
    def show_info(self, show='count'):
        
        if show == 'count':
            print(self.h_df['quadrant'].value_counts()) 
        self.h_df.to_excel(f'df{self.args.dataset}{self.args.split_class}.xlsx')

        
    def _collect_h_info(self, sample_num=300):
        print('Collect hidden vector - h space information ......')
        output_dict = {'target' : [], 
                       'base' : [], 
                       'target_idx' : [],
                       'base_idx' : [],
                       'kld' : [], 
                       'mse' : []}
        
        modelConfig = self.model_config

        with torch.no_grad():
            device = torch.device(modelConfig["device"])
            model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
            ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.")
            model.eval()
            sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        
        target_len = len(self.h_info[f'[{self.base}]'][1])

        # For the same noise
        noisyImage_for_mse = torch.randn(
            size=[1, 3, 32, 32], device=device)
        
        for i in range(sample_num):
            if i % 100 == 0 :
                print(f'Sample number : {i+1} / {sample_num}')
            rd_1 = random.randint(1, target_len - 1)
            rd_2 = random.randint(1, target_len - 1)
            
            for key, value in self.h_info.items():
                out_w_mse = 0
                out_w_kld = 0
                curr_step = 1000

                temp_input_h = []
                temp_target_h = []
                for i in range(0, len(self.h_info)):
                    temp_input_h.append(self.h_info[f'{key}'][i][rd_1:rd_1+1])
                    temp_target_h.append(self.h_info[f'[{self.base}]'][i][rd_2:rd_2+1])
                
                for t in range(1, 4):
                #for t in range(1, len(self.h_info)):
                    curr_step -= 100
                    
                    curr_input = self.h_info[f'{key}'][t] # T = 700
                    curr_target = self.h_info[f'[{self.base}]'][t] # T = 300
                    
                    kl_loss = nn.KLDivLoss(reduction='batchmean')
                    kld_input = F.log_softmax(curr_input[rd_1], dim=1)
                    kld_target = F.softmax(curr_target[rd_2], dim=1)
                    out_w_kld += round(kl_loss(kld_input, kld_target).item(), 4) * np.log(curr_step / 1000 + 1)
                    
                    """Algo 2 - not sure"""
                    """
                    #_mse_input = self.h_info[f'{key}'][t][rd_1:rd_1+1]
                    #_mse_target = self.h_info[f'[{self.base}]'][t][rd_2:rd_2+1]
                    #mse_input = self.sampler.model.sample_from_h_space(self.xt[t][0:1], torch.Tensor([t*100-1]).to(int).to('cuda'), _mse_input)
                    #mse_target = self.sampler.model.sample_from_h_space(self.xt[t][0:1], torch.Tensor([t*100-1]).to(int).to('cuda'), _mse_target)
                                            
                    with torch.no_grad():
                        sampler_input = GaussianDiffusionSampler(
                        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], temp_input_h, timestep_edition=t).to(device)
                        # Sampled from standard normal distribution
                        
                        input_sample = sampler_input(noisyImage_for_mse)
                        mse_input = input_sample * 0.5 + 0.5  # [0 ~ 1]

                        sampler_target = GaussianDiffusionSampler(
                        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], temp_target_h, timestep_edition=t).to(device)
                        # Sampled from standard normal distribution
                        
                        target_sample = sampler_target(noisyImage_for_mse)
                        mse_target = target_sample * 0.5 + 0.5  # [0 ~ 1]
                    """

                    with torch.no_grad():
                        _mse_input = self.h_info[f'{key}'][t][rd_1:rd_1+1]
                        _mse_target = self.h_info[f'[{self.base}]'][t][rd_2:rd_2+1]
                        mse_input = sampler.model.sample_from_h_space(self.xt[t][0:1], torch.Tensor([t*100-1]).to(int).to('cuda'), _mse_input)
                        mse_target = sampler.model.sample_from_h_space(self.xt[t][0:1], torch.Tensor([t*100-1]).to(int).to('cuda'), _mse_target)
                       
                    #mse_input = curr_input[rd_1]
                    #mse_target = curr_target[rd_2]

                    #_mse_input = self.h_info[f'{key}'][t][rd_1:rd_1+1]
                    #_mse_target = self.h_info[f'[{self.base}]'][t][rd_2:rd_2+1]
                    #mse_input = self.model.sample_from_h_space(self.xt[t][0:1], torch.Tensor([t*100-1]).to(int).to('cuda'), _mse_input)
                    #mse_target = self.model.sample_from_h_space(self.xt[t][0:1], torch.Tensor([t*100-1]).to(int).to('cuda'), _mse_target)
                    
                    mse_loss = nn.MSELoss(reduction='sum')
                    out_w_mse += mse_loss(mse_input, mse_target).item() * np.log(curr_step / 1000 + 1)

                output_dict['target'].append(key)
                output_dict['base'].append(self.base)
                output_dict['target_idx'].append(rd_1)
                output_dict['base_idx'].append(rd_2)
                output_dict['kld'].append(out_w_kld)
                output_dict['mse'].append(out_w_mse)
                
        output_df = self._calculate_boundary(pd.DataFrame.from_dict(output_dict))
                
        return output_df
          
    def _calculate_boundary(self, df):
        x_boundary = df[df['target'] == f'[{self.base}]'][['mse']].quantile(0.95).iloc[0]
        y_boundary = df[df['target'] == f'[{self.base}]'][['kld']].quantile(0.95).iloc[0]
        print(f'{self.args.dataset} label {self.args.split_class}\n x-axis boundary : {x_boundary}\n y-axis boundary : {y_boundary}')
        
        condition = [
        (df['mse'] > x_boundary) & (df['kld'] > y_boundary),
        (df['mse'] < x_boundary) & (df['kld'] > y_boundary),
        (df['mse'] < x_boundary) & (df['kld'] < y_boundary),
        (df['mse'] > x_boundary) & (df['kld'] < y_boundary),
        ]
        choices = ['1','2','3','4']
        df['quadrant'] = np.select(condition,choices)
        
        return df
    
    def _collect_xt(self, modelConfig):
        xt_list = []
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 3, 32, 32], device=device)

        sampler = GaussianDiffusionSampler(
                    model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        with torch.no_grad():
            x_t = noisyImage
            for time_step in reversed(range(1000)):
                t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
                if (time_step+1) % 100 == 0: #(1000 - self.timestep_edition * 100) :
                    print(f'Adding xt = {time_step}......')
                    #print(f'Adding xt shape = {x_t.shape}......')

                    xt_list.append(x_t)
                    #new_h = self.h_space_edition[self.timestep_edition]
                    #x_t = self.model.sample_from_h_space(x_t, t, new_h)

                mean, var= sampler.p_mean_variance(x_t=x_t, t=t)
                # no noise when t == 0
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.sqrt(var) * noise
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        return xt_list
    
class anchor_analysis:
    def __init__(self, model_config, args, base=0):
        #self.model = model
        #self.sampler = sampler
        self.args = args
        #self.xt = self._collect_xt(model_config)
        self.model_config = model_config
        #self.h_info = h_info
        #self.xt_info = xt_info 
        self.base = base
        self.anchor_df = pd.DataFrame()
        #self.h_df = pd.DataFrame.from_dict(self.h_dict)
        self.x_boundary = 0
        self.y_boundary = 0
        self._collect_data_intrinsic_and_extrinsic(args=self.args)

    def plot_df(self):
        orders = ['0','1','2','3','4','5','6','7','8','9']
        plt.figure(figsize=(16,9))
        sns.scatterplot(data=self.anchor_df, x='extrinsic', y='intrinsic', hue='label', hue_order=orders)
        plt.title(f'Scatter Plot - Different Classes Compare with {self.args.dataset}{self.args.split_class}', fontsize = 18)
        plt.savefig(f'Result/hInfo{self.args.dataset}/scatter_plot_anchor_label{self.args.dataset}_{self.args.split_class}.png')

        plt.figure(figsize=(16,9))
        sns.scatterplot(data=self.anchor_df, x='extrinsic', y='intrinsic', hue='quadrant')
        plt.title(f'Scatter Plot - Different Classes Compare with {self.args.dataset}{self.args.split_class}', fontsize = 18)
        plt.savefig(f'Result/hInfo{self.args.dataset}/scatter_plot_anchor_quadrant{self.args.dataset}_{self.args.split_class}.png')

        #plt.figure(figsize=(16,9))
        #h_df = self.h_df
        #ax = sns.scatterplot(data=h_df, x='mse', y='kld', hue='target')
        #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        #plt.title(f'Scatter Plot - Different Classes Compare with {self.args.dataset}{self.args.split_class}', fontsize = 18)
        #plt.savefig(f'Result/hInfo{self.args.dataset}/scatter_plot_target_{self.args.dataset}_{self.args.split_class}.png')

        #plt.figure(figsize=(16,9))
        #h_df = self.h_df
        #ax = sns.scatterplot(data=h_df, x='mse', y='kld', hue='quadrant')
        #sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        #plt.title(f'Scatter Plot - Different Quadrant Compare with {self.args.dataset}{self.args.split_class}', fontsize = 18)
        #plt.savefig(f'Result/hInfo{self.args.dataset}/scatter_plot_quadrant_{self.args.dataset}_{self.args.split_class}.png')
        
    def show_info(self, show='count'):
        with sns.color_palette('deep'):
            if show == 'count':
                print(self.anchor_df['quadrant'].value_counts()) 
                self.anchor_df.to_excel(f'df{self.args.dataset}{self.args.split_class}.xlsx')
                sns.catplot(
                data=self.anchor_df, x="quadrant",col="label",
                kind="count", height=4, aspect=.6, order=['1','2','3','4'], hue='is_train'
            )
                plt.title(f'Bar Plot - Different Classes Compare with {self.args.dataset}{self.args.split_class}', fontsize = 18)
                plt.savefig(f'Result/hInfo{self.args.dataset}/bar_plot_anchor_quadrant{self.args.dataset}_{self.args.split_class}.png')

                sns.boxplot(
                data=self.anchor_df, x="label", y="extrinsic", hue='is_train'
               
            )
                plt.title(f'Box Plot - Different Classes Compare with {self.args.dataset}{self.args.split_class}', fontsize = 18)
                plt.savefig(f'Result/hInfo{self.args.dataset}/box_plot_anchor_extrinsict{self.args.dataset}_{self.args.split_class}.png')
                
                sns.boxplot(
                data=self.anchor_df, x="label", y="intrinsic", hue='is_train'
               
            )
                plt.title(f'Box Plot - Different Classes Compare with {self.args.dataset}{self.args.split_class}', fontsize = 18)
                plt.savefig(f'Result/hInfo{self.args.dataset}/box_plot_anchor_intrinsict{self.args.dataset}_{self.args.split_class}.png')

    def _add_noise(self, img):
        row,col,ch= img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy_img = img + gauss

        return noisy_img.to(torch.float32)

    def _collect_data_intrinsic_and_extrinsic_v2(self, args, intrinsic_step=801, compared_intrinsic_step=301, extrinsic_step=31, compared_extrinsic_step=1):
        num_iter = args.num_iter

        dt = CUSTOM_DATASET(self.args, split=False)
        train_dataset, test_dataset = dt.load_dataset(custom_trasform=False)

        with torch.no_grad():
            device = torch.device(self.model_config["device"])
            model = UNet(T=self.model_config["T"], ch=self.model_config["channel"], ch_mult=self.model_config["channel_mult"], attn=self.model_config["attn"],
                    num_res_blocks=self.model_config["num_res_blocks"], dropout=0.)
            ckpt = torch.load(os.path.join(
            self.model_config["save_weight_dir"], self.model_config["test_load_weight"]), map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.")
            model.eval()
            
            trainer = GaussianDiffusionTrainer(
                    model, self.model_config["beta_1"], self.model_config["beta_T"], self.model_config["T"]).to(device)
            sampler = GaussianDiffusionSampler(
            model, self.model_config["beta_1"], self.model_config["beta_T"], self.model_config["T"]).to(device)
            # Sampled from standard normal distribution
            noisyImage = torch.randn(
            size=[self.model_config["batch_size"], 3, 32, 32], device=device)
            saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
            sampledImgs_base = sampler(noisyImage)
            sampledImgs_base = sampledImgs_base * 0.5 + 0.5  # [0 ~ 1]

        with torch.no_grad():

            res_n = []
            res_label = []
            res_intrinsic = []
            res_extrinsic = []
            res_istrain = []
            for n, (img, label) in enumerate(train_dataset):
                curr_step = extrinsic_step
                compare_step = compared_extrinsic_step
                #img = self._add_noise(img)
                curr_input = img.to('cuda')
                
                res_h = 0
                res_xt = 0
                for num in range(num_iter):
                    img_num = num
                    sample_base = sampledImgs_base[img_num]
                    base = trainer.get_step_xt(sample_base.to('cuda'), curr_step)
                    compared_base = trainer.get_step_xt(base.to('cuda'), compare_step)

                    base_n_t = base.new_ones([1, ], dtype=torch.long) * curr_step
                    compared_base_n_t = base.new_ones([1, ], dtype=torch.long) * compare_step


                    
                    #n_t = curr_input.new_ones([1, ], dtype=torch.long) * (curr_step + 1)

                    ### new 
                    base_n_t_h = base.new_ones([1, ], dtype=torch.long) * intrinsic_step

                    base_to_model = trainer.get_step_xt(sample_base.to('cuda'), intrinsic_step)
                    base_h = model.get_h_space(base_to_model.unsqueeze(dim=0).to(device), base_n_t_h.to(device))
                    compared_base_n_t_h = base.new_ones([1, ], dtype=torch.long) * compared_intrinsic_step
                    ### 

                    curr_xt_input = trainer.get_step_xt(curr_input.to('cuda'), curr_step)#.clip(base.min(),1)
                    compared_curr_xt_input = trainer.get_step_xt(curr_input.to('cuda'), compare_step)
                    #curr_xt_input = curr_input
                    compared_curr_input_to_model = trainer.get_step_xt(curr_input.to('cuda'), compared_intrinsic_step)
                    curr_input_to_model = trainer.get_step_xt(curr_input.to('cuda'), intrinsic_step)
                    curr_h_input = model.get_h_space(curr_input_to_model.unsqueeze(dim=0).to(device), base_n_t.to(device))#.clip(base.min(),1).unsqueeze(dim=0).to(device), base_n_t.to(device))
                    compared_curr_h_input = model.get_h_space(compared_curr_input_to_model.unsqueeze(dim=0).to(device), compared_base_n_t_h.to(device))

                    base = model(base.unsqueeze(dim=0), base_n_t)
                    #curr_xt_input = model(curr_xt_input.unsqueeze(dim=0), base_n_t)
                    #base_h = F.log_softmax(base_h)
                    #curr_h_input = F.log_softmax(curr_h_input)

                    mse_loss = nn.MSELoss(reduction='sum')
                    """
                    res_xt += mse_loss(base[0], curr_xt_input[0]).item()#* np.log(curr_step / 1000 + 1)
                    res_h += mse_loss(base_h, curr_h_input).item()
                    """
                    res_xt += mse_loss(base[0], compared_curr_xt_input[0]).item()#* np.log(curr_step / 1000 + 1)
                    res_h += mse_loss(base_h, compared_curr_h_input).item()


                    #print(f'extrinsic loss : {round(res_xt, 4)}, intrinsic loss : {round(res_h, 4)} label : {label}')

                res_n.append(n)
                res_label.append(str(label))
                res_intrinsic.append(round(res_h / 5, 4))
                res_extrinsic.append(round(res_xt / 5, 4))

                if label == self.args.base:
                    res_istrain.append('Train')
                else:
                    res_istrain.append('Test')

                if n == 500:
                    break
            
            img_df = pd.DataFrame([res_n, res_label, res_extrinsic, res_intrinsic, res_istrain]).T
            img_df.columns = ['num', 'label', 'extrinsic', 'intrinsic', 'is_train']
            self.anchor_df = self._calculate_boundary(img_df).sort_values(by=['label'])

    def _collect_data_intrinsic_and_extrinsic(self, args):


        num_iter = args.num_iter

        dt = CUSTOM_DATASET(self.args, split=False)
        train_dataset, test_dataset = dt.load_dataset(custom_trasform=False)

        with torch.no_grad():
            device = torch.device(self.model_config["device"])
            model = UNet(T=self.model_config["T"], ch=self.model_config["channel"], ch_mult=self.model_config["channel_mult"], attn=self.model_config["attn"],
                    num_res_blocks=self.model_config["num_res_blocks"], dropout=0.)
            ckpt = torch.load(os.path.join(
            self.model_config["save_weight_dir"], self.model_config["test_load_weight"]), map_location=device)
            model.load_state_dict(ckpt)
            print("model load weight done.")
            model.eval()
            
            trainer = GaussianDiffusionTrainer(
                    model, self.model_config["beta_1"], self.model_config["beta_T"], self.model_config["T"]).to(device)
            sampler = GaussianDiffusionSampler(
            model, self.model_config["beta_1"], self.model_config["beta_T"], self.model_config["T"]).to(device)
            # Sampled from standard normal distribution
            noisyImage = torch.randn(
            size=[self.model_config["batch_size"], 3, 32, 32], device=device)
            saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
            sampledImgs_base = sampler(noisyImage)
            sampledImgs_base = sampledImgs_base * 0.5 + 0.5  # [0 ~ 1]

        with torch.no_grad():

            res_n = []
            res_label = []
            res_intrinsic = []
            res_extrinsic = []
            res_istrain = []
            for n, (img, label) in enumerate(train_dataset):
                curr_step = 1
                img = self._add_noise(img)
                curr_input = img.to('cuda')
                
                res_h = 0
                res_xt = 0
                for num in range(num_iter):
                    img_num = num
                    base = sampledImgs_base[img_num]
                    base_n_t = base.new_ones([1, ], dtype=torch.long) * curr_step
                    base_h = model.get_h_space(base.unsqueeze(dim=0).to(device), base_n_t.to(device))
                    n_t = curr_input.new_ones([1, ], dtype=torch.long) * (curr_step + 1)


                    curr_xt_input = trainer.get_step_xt(curr_input.to('cuda'), curr_step)#.clip(base.min(),1)
                    #curr_xt_input = curr_input
                    curr_h_input = model.get_h_space(curr_input.unsqueeze(dim=0).to(device), base_n_t.to(device))#.clip(base.min(),1).unsqueeze(dim=0).to(device), base_n_t.to(device))

                    #base_h = F.log_softmax(base_h)
                    #curr_h_input = F.log_softmax(curr_h_input)

                    mse_loss = nn.MSELoss(reduction='sum')
                    res_xt += mse_loss(base[0], curr_xt_input[0]).item()#* np.log(curr_step / 1000 + 1)
                    res_h += mse_loss(base_h, curr_h_input).item()
                    #print(f'extrinsic loss : {round(res_xt, 4)}, intrinsic loss : {round(res_h, 4)} label : {label}')

                res_n.append(n)
                res_label.append(str(label))
                res_intrinsic.append(round(res_h / 5, 4))
                res_extrinsic.append(round(res_xt / 5, 4))

                if label == self.args.base:
                    res_istrain.append('Train')
                else:
                    res_istrain.append('Test')

                if n == 2500:
                    break
            
            img_df = pd.DataFrame([res_n, res_label, res_extrinsic, res_intrinsic, res_istrain]).T
            img_df.columns = ['num', 'label', 'extrinsic', 'intrinsic', 'is_train']
            self.anchor_df = self._calculate_boundary(img_df).sort_values(by=['label'])


    def _calculate_boundary(self, df):
        x_boundary = df[df['label'] == f'{self.base}'][['extrinsic']].quantile(0.95).iloc[0]
        y_boundary = df[df['label'] == f'{self.base}'][['intrinsic']].quantile(0.95).iloc[0]
        print(f'{self.args.dataset} label {self.args.split_class}\n x-axis boundary : {x_boundary}\n y-axis boundary : {y_boundary}')
        
        condition = [
        (df['extrinsic'] > x_boundary) & (df['intrinsic'] > y_boundary),
        (df['extrinsic'] < x_boundary) & (df['intrinsic'] > y_boundary),
        (df['extrinsic'] < x_boundary) & (df['intrinsic'] < y_boundary),
        (df['extrinsic'] > x_boundary) & (df['intrinsic'] < y_boundary),
        ]
        choices = ['1','2','3','4']
        df['quadrant'] = np.select(condition,choices)
        
        return df
    

def model_init(args, model_config, rd=False):
    """
    Initiate the DDPM model, sampler. and trainer.
    Args:
        args: (Unused currently)
        model_config: Containing model hyperparameters and configurations
        rd: Passing the reduced model arguments

    Returns:
        model: DDPM model 
        sampler: DDPM Gaussian diffusion sampler
        trainer: DDPM trainer
    
    """
    # Laod reduced model instead
    if rd == True:
        args.load_different_model = True


    modelConfig = set_config(args)
    print(f"loading model : {modelConfig['training_load_weight']} ......")

    # Need CUDA to start 
    device = torch.device(modelConfig["device"])

    
    noisyImage = torch.randn(
        size=[modelConfig["batch_size"], 3, 32, 32], device=device)


    model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                    num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
    
    ckpt = torch.load(os.path.join(
    modelConfig["load_weight_dir"], modelConfig["test_load_weight"]), map_location=device)

    model.load_state_dict(ckpt)

    print("model load weight done.")
    model.eval()

    sampler = GaussianDiffusionSampler(
        model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    
    print("update Sampler ...... done")

    trainer = GaussianDiffusionTrainer(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    
    print("update Trainer ...... done")

    args.load_different_model = False

    return model, sampler, trainer


def create_reduce_idx(args, df, e_w=0.2, i_w=0.8):
    df_copy = df.copy()
    #df_anchor_is_train = df_is_train[df_is_train['is_train']==True]
    df_anchor_mean = df_copy.groupby(['idx'])[['scaled_ef_inversion','scaled_h']].mean().reset_index()
    df_anchor_mean['weight'] = (df_anchor_mean['scaled_ef_inversion'] * e_w + df_anchor_mean['scaled_h'] * i_w)
    
    df_anchor_mean = df_anchor_mean.sort_values(by='weight')
    reduced_sort_idx = np.array(df_anchor_mean['idx'])

    return reduced_sort_idx

def load_rd_model(args, modelConfig):
    args.load_different_model = True
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)

        """Modify in different enviornment"""
        ckpt = torch.load(os.path.join(
            modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
    
    return model, sampler

def generate_sameple_img(args, model_config, model, sampler):

    """
    Some bugs need to be fixed here :
    sampling time became longer after training process.
    after gc.collect() still doesn't work.
    """
    with torch.no_grad():
        model.eval()
        noisyImage = torch.randn(
        size=[model_config["batch_size"], 3, 32, 32], device="cuda")
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        sampledImgs = sampler(noisyImage)

    return sampledImgs