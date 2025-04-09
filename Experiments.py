import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

        """Initialize the config for the procedure"""
        self.test_t = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 999]

        """Merge with _procedure function"""
        # None stands for unlimit
        self.test_lim_i = 80 
        self.train_lim_i = 80


        # Run
        print("Creating anchor dataframe ......")
        self.df_anchor = self._procedure(model, sampler, trainer, args.training_anchor_num, args.testing_anchor_num)
        print("Create anchor dataframe Done ......")

    def _procedure(self, model, sampler, trainer, training_anchor_num=10, testing_anchor_num=10):
        """
        Still need some improvment for divided extrinsic and intrinsic 't'
        """
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
        
        #Function for adjusting weight to applying time step
        test_t_ext_w = 0
        test_t_int_w = 1

        with torch.no_grad():
            model.eval()
            noisyImage = torch.randn(
            size=[self.model_config["batch_size"], 3, 32, 32], device=self.device)
            saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
            sampledImgs = sampler(noisyImage)

        x_tilde_0 = sampledImgs

        dt_train = CUSTOM_DATASET(self.args, split=True)
        train_dataset, _ = dt_train.load_dataset(custom_trasform=False)
        dt = CUSTOM_DATASET(self.args, split=False)
        _, test_dataset = dt.load_dataset(custom_trasform=False)


        with torch.no_grad():
            for test_t in self.test_t:
                test_t_ext = int(test_t * test_t_ext_w)
                test_t_int = int(test_t * test_t_int_w) 

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
                    ef_inversion_val = edit_friendly_inversion(x_0_train, test_t_ext, alpha_cumprod, model, sampler)
                
                    # h space
                    base_n_t_h = x_0_train.new_ones([1, ], dtype=torch.long) * test_t_int
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
                    ef_inversion_val = edit_friendly_inversion(x_0, test_t_ext, alpha_cumprod, model, sampler)

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
                    
                    
                    if i == testing_anchor_num:
                        break


                for i, (img, label) in enumerate(train_dataset):
                    x_0 = train_dataset[i][0].to(self.device)
                    #x_tilde_t = trainer.get_step_xt(x_tilde_0, test_t)
                    #timesteps = list(range(1000))
                    ef_inversion_val = edit_friendly_inversion(x_0, test_t_ext, alpha_cumprod, model, sampler)

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
                    
                    
                    if i == training_anchor_num:
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

        fig, axes = plt.subplots(2, 1, figsize=(6, 12))  # Adjust figure size as needed

        sns.lineplot(ax=axes[0], data=self.df_anchor, x='t',y='scaled_ef_inversion',hue='label', hue_order=hue_order, palette=palette)
        axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))  # Moves the legend to the right
        axes[0].set_title(f'Extrinsic at each timestep{self.args.dataset,self.args.split_class}')
        plt.savefig(f'Result/Results{self.args.dataset}/ef_inversion_{self.args.dataset}_{self.args.split_class}.png')

        sns.lineplot(ax=axes[1], data=self.df_anchor, x='t',y='scaled_h',hue='label', hue_order=hue_order, palette=palette)
        axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))  # Moves the legend to the right
        axes[1].set_title(f'Intrinsic at each timestep{self.args.dataset,self.args.split_class}')
        plt.savefig(f'Result/Results{self.args.dataset}/scaled_h_{self.args.dataset}_{self.args.split_class}.png')


class Experiment_2:
    """Calculate Frechet Distance"""
    def __init__(self, args, model_config, experiment_1_df, metric):
        self.args = args
        self.metric = metric
        self.exp1_df = experiment_1_df.copy()
        df_istrain = experiment_1_df[experiment_1_df['is_train'] == True]
        df_istest = experiment_1_df[experiment_1_df['is_train'] == False]
        
        (
            self.anchor_train_mean_df,
            self.i_mean,
            self.i_var,
            self.e_mean,
            self.e_var
        ) = self._calc_asc_weight_df(df_istrain, df_istrain, metric=metric)
        
        self.anchor_test_mean_df, _, _, _, _ = self._calc_asc_weight_df(df_istest, df_istrain,metric=metric)
        
   
    
    def plot_sim(self):
        args = self.args

        palette = {
            str(i): 'red' if str(i) == str(args.split_class[0]) else 'grey'
            for i in range(10)
        }
        palette['train'] = 'blue'  # Adding 'train' key separately
        hue_order = ['0','1','2','3','4','5','6','7','8','9','train']

        concat_df = pd.concat([self.anchor_train_mean_df, self.anchor_test_mean_df],axis=0).reset_index()

        fig, axes = plt.subplots(3, 1, figsize=(6, 18))  # Adjust figure size as needed

        sns.boxplot(ax=axes[0], data=concat_df, x='label',y='fd_h', hue_order=hue_order, palette=palette, order=hue_order)
        axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))  # Moves the legend to the right
        axes[0].set_title(f'{self.metric} distance of scaled intrinsic{args.dataset,args.split_class}')
        plt.savefig(f'Result/Results{self.args.dataset}/{self.metric}_h_{self.args.dataset}_{self.args.split_class}.png')


        #hue_order = ['0','1','2','3','4','5','6','7','8','9']
        hue_order = ['0','1','2','3','4','5','6','7','8','9','train']

        sns.boxplot(ax=axes[1], data=concat_df, x='label',y='fd_ef', hue_order=hue_order, palette=palette, order=hue_order)
        axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))  # Moves the legend to the right
        axes[1].set_title(f'{self.metric} distance of scaled extrinsic{args.dataset,args.split_class}')
        plt.savefig(f'Result/Results{self.args.dataset}/{self.metric}_ex_{self.args.dataset}_{self.args.split_class}.png')

        sns.scatterplot(data=concat_df, x="fd_ef", y="fd_h", hue="label", palette=palette)
        axes[2].legend(loc='upper left', bbox_to_anchor=(1, 1))  # Moves the legend to the right
        axes[2].set_title(f'Quadrant plot of anchor {args.dataset,args.split_class}')
        plt.savefig(f'Result/Results{self.args.dataset}/scatter_plot_{self.metric}_{self.args.dataset}_{self.args.split_class}.png')

        
    def _calc_asc_weight_df(self, input_df, compared_df, set_compare_num=80, metric='L2'):
        fd_h_lst = []
        fd_ef_lst = []
        label_lst = []
        compare_t = input_df['t'].unique()

        #num_train = len(input_df['idx'].unique())

        #calculate training data
        for i in input_df['idx'].unique():
            fd_h = 0
            fd_ef = 0

            df_istrain_i = input_df[input_df['idx']==i]
            is_train_i_h_arr = np.array(df_istrain_i['scaled_h'])
            is_train_i_ef_arr = np.array(df_istrain_i['scaled_ef_inversion'])
            compare_h_P = np.vstack([compare_t, is_train_i_h_arr]).T
            compare_ef_P = np.vstack([compare_t, is_train_i_ef_arr]).T

            for counter, j in enumerate(compared_df['idx'].unique()):
                if counter > set_compare_num:
                    break
                
                df_istrain_j = compared_df[compared_df['idx']==j]
                is_train_j_h_arr = np.array(df_istrain_j['scaled_h'])
                is_train_j_ef_arr = np.array(df_istrain_j['scaled_ef_inversion'])
                compare_h_Q = np.vstack([compare_t, is_train_j_h_arr]).T
                compare_ef_Q = np.vstack([compare_t, is_train_j_ef_arr]).T
                
                fd_h += self._distance_metric(compare_h_P, compare_h_Q, metric=metric) / set_compare_num
                fd_ef += self._distance_metric(compare_ef_P, compare_ef_Q, metric=metric) / set_compare_num


            fd_h_lst.append(fd_h)
            fd_ef_lst.append(fd_ef)
            label_lst.append(input_df.iloc[i].label)
            
        i_star = np.argmin(fd_h_lst)
        self._save_i_starImg(i_star)
         #calculate training data
        for i in [i_star]:
            i_star_i = []
            i_star_e = []

            df_istrain_i = input_df[input_df['idx']==i]
            is_train_i_h_arr = np.array(df_istrain_i['scaled_h'])
            is_train_i_ef_arr = np.array(df_istrain_i['scaled_ef_inversion'])
            compare_h_P = np.vstack([compare_t, is_train_i_h_arr]).T
            compare_ef_P = np.vstack([compare_t, is_train_i_ef_arr]).T


            for counter, j in enumerate(compared_df['idx'].unique()):
                if counter > set_compare_num:
                    break
                
                df_istrain_j = compared_df[compared_df['idx']==j]
                is_train_j_h_arr = np.array(df_istrain_j['scaled_h'])
                is_train_j_ef_arr = np.array(df_istrain_j['scaled_ef_inversion'])
                compare_h_Q = np.vstack([compare_t, is_train_j_h_arr]).T
                compare_ef_Q = np.vstack([compare_t, is_train_j_ef_arr]).T

                fd_h += self._distance_metric(compare_h_P, compare_h_Q, metric=metric) / set_compare_num
                fd_ef += self._distance_metric(compare_ef_P, compare_ef_Q, metric=metric) / set_compare_num

                
                i_star_i.append(self._distance_metric(compare_h_P, compare_h_Q, metric=metric))
                i_star_e.append(self._distance_metric(compare_ef_P, compare_ef_Q, metric=metric))
    
        i_mean = np.mean(i_star_i)
        i_var = np.var(i_star_i)
        e_mean = np.mean(i_star_e)
        e_var = np.var(i_star_e)

        input_df_mean = input_df.groupby(['idx','label'])[['scaled_h','scaled_ef_inversion']].mean().reset_index()
        fd_df = pd.DataFrame([fd_h_lst, fd_ef_lst]).T
        fd_df.columns = ['fd_h','fd_ef']
        anchor_mean_df = pd.concat([input_df_mean,fd_df],axis=1)

        return anchor_mean_df, i_mean, i_var, e_mean, e_var

    def _euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _frechet_recursive(self, ca, P, Q, i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = self._euclidean(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(self._frechet_recursive(ca, P, Q, i-1, 0), self._euclidean(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(self._frechet_recursive(ca, P, Q, 0, j-1), self._euclidean(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(self._frechet_recursive(ca, P, Q, i-1, j),
                            self._frechet_recursive(ca, P, Q, i-1, j-1),
                            self._frechet_recursive(ca, P, Q, i, j-1)),
                        self._euclidean(P[i], Q[j]))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    def _frechet_distance(self, P, Q):
        ca = np.ones((len(P), len(Q))) * -1
        return self._frechet_recursive(ca, P, Q, len(P)-1, len(Q)-1)
    
    def _distance_metric(self, P, Q, metric):
        """
        Compute the distance between two sequences using various distance metrics.

        Parameters:
            P (np.array): First sequence
            Q (np.array): Second sequence
            metric (str): The distance metric to use. Options:
                        - 'L1' (Manhattan Distance)
                        - 'L2' (Euclidean Distance)
                        - 'Linf' (Chebyshev Distance)
                        - 'Frechet' (Frechet Distance)

        Returns:
            float: Computed distance
        """
        P, Q = np.array(P), np.array(Q)  # Ensure NumPy arrays
        
        if metric == 'L1':  # Manhattan Distance
            return np.sum(np.abs(P - Q))
        
        elif metric == 'L2':  # Euclidean Distance
            return np.sqrt(np.sum((P - Q) ** 2))
        
        elif metric == 'Linf':  # Chebyshev Distance (Maximum Norm)
            return np.max(np.abs(P - Q))
        
        elif metric == 'Frechet':  # Frechet Distance (assuming implemented)
            return self._frechet_distance(P, Q)
        
        else:
            raise ValueError(f"Unknown metric: {metric}. Choose from 'L1', 'L2', 'Linf', or 'Frechet'.")
    
    def _save_i_starImg(self, i_star):
        dt_train = CUSTOM_DATASET(self.args, split=True)
        train_dataset, _ = dt_train.load_dataset(custom_trasform=True)
        #plt.imshow(train_dataset[i_star][0].permute(1,2,0))
        #plt.savefig(f'Result/Results{self.args.dataset}/istar_{self.args.metric}_{self.args.dataset}_{self.args.split_class}.png')


class Experiment_3:
    def __init__(self, args, model_config, exp_train_df, exp_test_df):
        self.args = args
        self.df_train = exp_train_df.copy()
        self.df_test = exp_test_df.copy()
        
    def plot(self, KDE_plot=False):
        self._procedure(self.df_train, self.df_test, KDE_plot)  
        
    def _procedure(self, df_train, df_test, KDE_plot):
        # Define the palette
        args = self.args
        
        palette = {
            str(i): 'red' if str(i) == str(args.split_class[0]) else 'grey'
            for i in range(10)
        }
        palette['train'] = 'blue'  # Adding 'train' key separately

        hue_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'train']

        # Concatenating DataFrames
        concat_df = pd.concat([df_train, df_test], axis=0).reset_index()
        concat_df['w'] = concat_df['fd_h'] * 0.5 + concat_df['fd_ef'] * 0.5

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Boxplot
        sns.boxplot(ax=axes[0], data=concat_df, x='label', y='w', hue='label', order=hue_order, palette=palette)
        axes[0].set_title(f'Distance of data {args.dataset, args.split_class}')
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, title="Label", loc='upper left', bbox_to_anchor=(1, 1))  # Moves legend to the right

        # Histogram
        if KDE_plot:
            ax = sns.kdeplot(data=concat_df, x="w", hue='label', palette=palette)

            handles, labels = ax.get_legend_handles_labels()
            plt.title("Distribution of scaled_h for each label")
            plt.xlabel("scaled_h")
            plt.ylabel("Count")
            ax.legend(handles, labels, title="Label", loc='upper left', bbox_to_anchor=(1, 1))
            plt.show()
        else:
            sns.histplot(ax=axes[1], data=concat_df, x="w", hue="label", kde=True, bins=20, palette=palette)
            handles, labels = axes[1].get_legend_handles_labels()
            axes[1].legend(handles, labels, title="Label", loc='upper right')  # Adjust legend position
            axes[1].set_title("Distribution for each label")

            # Adjust labels
            axes[1].set_xlabel("scaled_h")
            axes[1].set_ylabel("Count")
            
        plt.savefig(f'Result/Results{self.args.dataset}/distribution_{self.args.metric}_{self.args.dataset}_{self.args.split_class}.png')
        plt.tight_layout()
        plt.show()
