import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn 
from torch.nn import functional as F

from sklearn.preprocessing import MinMaxScaler

from Metric import *
from Dataset import CUSTOM_DATASET

class Experiment_1:
    def __init__(self, args, model_config, model, sampler, trainer):
        self.args = args
        self.device = model_config['device']
        self.model_config = model_config

        # Run
        self.df_anchor = self._procedure(model, sampler, trainer)


    def _procedure(self, model, sampler, trainer):
        data_idx = []
        label_lst = []
        ef_inversion_lst = []
        h_lst = []
        t_lst = []
        kld_lst = []
        is_train = []


        mu_q_lst = []
        sigma_q_lst = []
        mu_p_lst = []
        sigma_p_lst = []

        betas = trainer.betas
        alpha_t = 1.0 - betas
        alpha_cumprod = torch.cumprod(alpha_t, dim=0)

        with torch.no_grad():
            model.eval()
            noisyImage = torch.randn(
            size=[self.model_config["batch_size"], 3, 32, 32], device=self.device)
            saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
            sampledImgs = sampler(noisyImage)

        x_tilde_0 = sampledImgs

        dt_train = CUSTOM_DATASET(self.args, split=True)
        train_dataset, _ = dt_train.load_dataset(custom_trasform=True)
        dt = CUSTOM_DATASET(self.args, split=False)
        _, test_dataset = dt.load_dataset(custom_trasform=True)


        with torch.no_grad():
            #for test_t in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 999]:
            #for test_t in [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]:
            #for test_t in [0, 50, 100, 150, 200, 250, 300]:
            for test_t in [20, 50]:

                """
                x_tilde_noise_map = []
                x_tilde_h_map = []
                #for n_batch in range(x_tilde_0.shape[0]):
                    #ef_inversion_val_tilde = edit_friendly_inversion(x_tilde_0[n_batch], test_t, alpha_cumprod)
                    #x_tilde_noise_map.append(ef_inversion_val_tilde)
                #x_tilde_noise_map = torch.stack(x_tilde_noise_map)

                for i, (train_img, train_label) in enumerate(train_dataset):
                    # noise 
                    x_0_train = train_dataset[i][0].to(device)
                    ef_inversion_val = edit_friendly_inversion(x_0_train, test_t, alpha_cumprod)
                    
                    # h space
                    base_n_t_h = x_0_train.new_ones([1, ], dtype=torch.long) * test_t
                    h_train = model(x_0_train.unsqueeze(dim=0), base_n_t_h)
                    
                    # append 
                    x_tilde_noise_map.append(ef_inversion_val)
                    x_tilde_h_map.append(h_train)
                    if i == 79:
                        break
                """
                x_tilde_noise_map = []
                x_tilde_h_map = []
                
                for i in range(len(x_tilde_0)):
                    x_0_train = x_tilde_0[i].to(self.device)
                    ef_inversion_val = edit_friendly_inversion(x_0_train, test_t, alpha_cumprod, model, sampler)
                
                    # h space
                    base_n_t_h = x_0_train.new_ones([1, ], dtype=torch.long) * test_t
                    h_train = model(x_0_train.unsqueeze(dim=0), base_n_t_h)
                    
                    # append 
                    x_tilde_noise_map.append(ef_inversion_val)
                    x_tilde_h_map.append(h_train)
                    if i == 79:
                        break
                
                x_tilde_noise_map = torch.stack(x_tilde_noise_map)
                x_tilde_h_map = torch.stack(x_tilde_h_map)
                
                for i, (img, label) in enumerate(test_dataset):
                    x_0 = test_dataset[i][0].to(self.device)
                    #x_tilde_t = trainer.get_step_xt(x_tilde_0, test_t)
                    #timesteps = list(range(1000))
                    ef_inversion_val = edit_friendly_inversion(x_0, test_t, alpha_cumprod, model, sampler)

                    h_test = model(x_0.unsqueeze(dim=0), base_n_t_h)
                    
                    #print(kl_div_pixelwise.mean().cpu().numpy(), test_dataset[i][1])
                    label_lst.append(str(label))
                    #kld_lst.append(float(kl_div_pixelwise.mean().cpu().numpy()))
                    #h_lst.append(h_diff)
                    t_lst.append(test_t)

                    mse_loss = nn.MSELoss()
                    kld_loss = nn.KLDivLoss(reduction='batchmean')

                    mse_diff = 0
                    h_diff = 0
                    kld_diff = 0

                    input_t = F.log_softmax(ef_inversion_val.unsqueeze(dim=0), dim=1)
                    target_t = F.softmax(x_tilde_noise_map, dim=1)
                    kld_diff = kld_loss(input_t, target_t)



                    for j in range(x_tilde_noise_map.shape[0]):
                        mse_diff += compute_edit_distance(ef_inversion_val * 100, x_tilde_noise_map[j] * 100, 'mse') / (x_tilde_noise_map.shape[0])
                        h_diff += compute_edit_distance(h_test, x_tilde_h_map[j], 'mse') / (x_tilde_noise_map.shape[0])
                    
                    ef_inversion_lst.append(round(float((mse_diff.cpu().numpy())), 4))
                    h_lst.append(round(float(h_diff.cpu().numpy()), 4))
                    kld_lst.append(round(float(kld_diff.cpu().numpy()), 4))
                    data_idx.append(int(i))
                    is_train.append(False)
                    
                    
                    ##mu_q_lst.append(round(float((mu_q.cpu().numpy())), 4))
                    #sigma_q_lst.append(round(float((sigma_q.cpu().numpy())), 4))
                    ##mu_p_lst.append(round(float((mu_p.cpu().numpy())), 4))
                    #sigma_p_lst.append(round(float((sigma_p.cpu().numpy())), 4))
                    
                    
                    if i == 25:
                        break


                for i, (img, label) in enumerate(train_dataset):
                    x_0 = train_dataset[i][0].to(self.device)
                    #x_tilde_t = trainer.get_step_xt(x_tilde_0, test_t)
                    #timesteps = list(range(1000))
                    ef_inversion_val = edit_friendly_inversion(x_0, test_t, alpha_cumprod, model, sampler)

                    h_test = model(x_0.unsqueeze(dim=0), base_n_t_h)
                    
                    #print(kl_div_pixelwise.mean().cpu().numpy(), test_dataset[i][1])
                    label_lst.append('train')
                    #kld_lst.append(float(kl_div_pixelwise.mean().cpu().numpy()))
                    #h_lst.append(h_diff)
                    t_lst.append(test_t)

                    mse_loss = nn.MSELoss()
                    kld_loss = nn.KLDivLoss(reduction='batchmean')

                    mse_diff = 0
                    h_diff = 0
                    kld_diff = 0

                    input_t = F.log_softmax(ef_inversion_val.unsqueeze(dim=0), dim=1)
                    target_t = F.softmax(x_tilde_noise_map, dim=1)
                    kld_diff = kld_loss(input_t, target_t)

                    for j in range(x_tilde_noise_map.shape[0]):
                        mse_diff += compute_edit_distance(ef_inversion_val * 100, x_tilde_noise_map[j] * 100, 'mse') / (x_tilde_noise_map.shape[0])
                        h_diff += compute_edit_distance(h_test, x_tilde_h_map[j], 'mse') / (x_tilde_noise_map.shape[0])
                    
                    ef_inversion_lst.append(round(float((mse_diff.cpu().numpy())), 4))
                    h_lst.append(round(float(h_diff.cpu().numpy()), 4))
                    kld_lst.append(round(float(kld_diff.cpu().numpy()), 4))
                    data_idx.append(int(i))
                    is_train.append(True)
                    
                    
                    if i == 79:
                        break

            
                
            df_anchor = pd.DataFrame([data_idx, is_train, label_lst, t_lst, ef_inversion_lst, h_lst, kld_lst]).T
            df_anchor.columns = ['idx', 'is_train', 'label', 't', 'ef_inversion', 'h', 'kld']
            
            # Create a MinMaxScaler instance
            scaler = MinMaxScaler()

            # Apply min-max scaling to ef_inversion within each group of t
            df_anchor['scaled_ef_inversion'] = df_anchor.groupby('t')['ef_inversion'].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
            )
            df_anchor['scaled_h'] = df_anchor.groupby('t')['h'].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
            )
            df_anchor['scaled_kld'] = df_anchor.groupby('t')['kld'].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
            )

            return df_anchor

    def plot_sim(self):
        palette = {
            str(i): 'red' if str(i) == str(self.args.split_class[0]) else 'grey'
            for i in range(10)
        }
        palette['train'] = 'blue'  # Adding 'train' key separately

        #hue_order = ['0','1','2','3','4','5','6','7','8','9']
        hue_order = ['0','1','2','3','4','5','6','7','8','9','train']

        sns.lineplot(data=self.df_anchor, x='t',y='ef_inversion',hue='label', hue_order=hue_order, palette=palette)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Moves the legend to the right
        plt.title(f'ef_inversion at each timestep{self.args.dataset,self.args.split_class}')
        plt.savefig(f'Result/{self.args.dataset}_{self.args.split_class}.png')