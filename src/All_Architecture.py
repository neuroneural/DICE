import torch
import torch.nn as nn
import torch.nn.utils.rnn as tn
from torch.autograd import Variable
import os
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.utils.data import TensorDataset
from src.auto_encoder_decoder import auto_encoder
from src.auto_encoder_decoder import auto_decoder
import scipy
import time
import copy
import random
from .utils import init
from polyssifier import poly
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class combinedModel(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(self, encoder, lstm, samples_per_subject, gain=0.1, PT="", exp="UFPT", device="cuda", oldpath="",k=10, n_regions=100,device_two=""):

        super().__init__()
        self.encoder = encoder

        self.lstm = lstm
        self.gain = gain
        # self.graph = graph
        self.samples_per_subject = 1
        self.n_clusters = 4
        self.auto_encoder = auto_encoder(self.samples_per_subject)
        self.auto_decoder = auto_decoder(self.samples_per_subject)
        self.w=1
        self.n_regions = n_regions
        self.n_regions_after = n_regions
        self.PT = PT
        self.exp = exp
        self.device = device
        self.device_two = device_two
        self.oldpath=oldpath
        self.time_points=155
        self.n_heads=4
        self.attention_embedding = 64 * self.n_heads
        self.k=10000#k
        self.upscale= .05#1##2#1#0.25 #HCP.005 and FBIRN region
        self.upscale2 = .5
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.temperature = 2

        # self.attn_spatial = nn.Sequential(
        #     nn.Linear(self.time_points, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1),
        # )


        # self.attn_weight = nn.Sequential(
        #     nn.Linear(self.n_regions*self.n_regions, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1),
        # )
        # self.attn_region = nn.Sequential(
        #     nn.Linear(self.lstm.output_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1),
        # )
        # self.attn_time = nn.Sequential(
        #     nn.Linear(self.lstm.output_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1),
        # )

        # self.lta_query = nn.Linear(self.n_regions * self.n_regions, self.n_regions * self.n_regions).to(self.device)
        # self.lta_key = nn.Linear(self.n_regions * self.n_regions, self.n_regions * self.n_regions).to(self.device)
        # self.lta_dropout = nn.Dropout(0.1)


        self.gta_embed = nn.Sequential(nn.Linear(self.n_regions * self.n_regions, round(self.upscale * self.n_regions * self.n_regions)),
                                        )#.to(self.device_two)

        self.gta_norm = nn.Sequential(nn.BatchNorm1d(round(self.upscale * self.n_regions * self.n_regions)), nn.ReLU())#.to(self.device_two)

        self.gta_attend = nn.Sequential(nn.Linear(round(self.upscale * self.n_regions * self.n_regions), round(self.upscale2 * self.n_regions * self.n_regions)),
                                         nn.ReLU(),
                                         nn.Linear(round(self.upscale2 * self.n_regions * self.n_regions), 1))#.to(self.device_two)


        # self.embed = nn.Linear(64,round(self.upscale2 * 64))  # .to(self.device_two)
        #
        # self.norm = nn.Sequential(nn.BatchNorm1d(round(self.upscale2 * 64)),
        #                                nn.GELU())  # .to(self.device_two)
        #
        # self.attend = nn.Linear(round(self.upscale2 * 64), 1)  # .to(self.device_two)



        # self.gta_embed_embeddings = nn.Linear(self.n_regions * self.attention_embedding,
        #                             round(self.upscale * self.n_regions * self.attention_embedding)).to(self.device)
        #
        # self.gta_norm_embeddings = nn.Sequential(nn.BatchNorm1d(round(self.upscale * self.n_regions * self.attention_embedding)),
        #                                nn.ReLU()).to(self.device)
        #
        # self.gta_attend_embeddings = nn.Linear(round(self.upscale * self.n_regions * self.attention_embedding), 1).to(self.device)
        #
        self.gta_dropout = nn.Dropout(0.35)

        # self.lta_query_components = nn.Linear(self.n_regions, self.n_regions)
        # self.lta_key_components = nn.Linear(self.n_regions, self.n_regions)
        # self.lta_dropout_components = nn.Dropout(0.1)



        # self.decoder = nn.Sequential(
        #     nn.Linear(self.lstm.output_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 2),
        #
        # )

        # self.mlp_before_lstm = nn.Sequential(
        #     nn.Linear(self.n_regions, self.n_regions),
        #     nn.ReLU(),
        #     nn.Linear(n_regions, self.n_regions_after),
        #     nn.ReLU(),
        #     # nn.Linear(64, self.n_regions_after),
        #     # nn.ReLU(),
        #     # nn.Linear(256, 100),
        #     # nn.ReLU(),
        #     # nn.Linear(128, 2)
        #
        # ).to(device)

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.n_regions_after * self.n_regions_after, 128),
        #     # nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     # nn.Linear(self.n_regions_after, self.n_regions_after),
        #     # nn.BatchNorm1d(64),
        #     # nn.Linear(2048, 2048),
        #     # nn.GELU(),
        #     nn.Linear(128,2),
        #     # nn.GELU(),
        #     # nn.ReLU6()
        #     # nn.Linear(116, 2)
        #     # nn.ReLU(),
        #     # nn.Linear(128, 2)
        #
        # ).to(self.device_two)

        # self.mlp2 = nn.Sequential(
        #     nn.Linear(self.n_regions_after * self.attention_embedding, 128),
        #     # nn.BatchNorm1d(512),
        #     nn.GELU(),
        #     # nn.Linear(512, 64),
        #     # nn.BatchNorm1d(64),
        #     # nn.GELU(),
        #     nn.Linear(128, 2),
        #     # nn.ReLU6()
        #     # nn.Linear(64, 2)
        #     # nn.ReLU(),
        #     # nn.Linear(128, 2)
        #
        # )  # .to(self.device)

        # self.mlp2 = nn.Sequential(
        #     nn.Linear(1, 16),
        #     nn.GELU(),
        #     nn.Linear(16, 16),
        #     nn.GELU(),
        #     nn.Linear(16, 1),
        #
        # )



        # self.lstm_decoder = nn.Sequential(
        #     nn.Linear(self.n_regions * self.lstm.output_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.attention_embedding)
        #
        # )
        #
        # self.lstm_decoder_time = nn.Sequential(
        #     nn.Linear(self.time_points * self.attention_embedding, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.attention_embedding)
        #
        # )
        #
        # self.lstm_decoder2 = nn.Sequential(
        #     nn.Linear(self.attention_embedding, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 2)
        #
        # )
        # self.lstm_decoder3 = nn.Sequential(
        #     nn.Linear(self.attention_embedding, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 2)
        #
        # )




        #self.encoder.feature_size
        #################################### MHA 1 ###############################################
        self.key_layer = nn.Sequential(
            nn.Linear(self.samples_per_subject * self.lstm.output_dim ,
                      self.samples_per_subject * self.attention_embedding),
        )#.to(self.device)

        self.value_layer = nn.Sequential(
            nn.Linear(self.samples_per_subject * self.lstm.output_dim ,
                      self.samples_per_subject * self.attention_embedding),
        )#.to(self.device)

        self.query_layer = nn.Sequential(
            nn.Linear(self.samples_per_subject * self.lstm.output_dim ,
                      self.samples_per_subject * self.attention_embedding),
        )#.to(self.device)

        # self.means_to_higher_projection = nn.Sequential(
        #     nn.Linear(64,256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.n_regions_after * self.n_regions_after),
        #
        # )  # .to(self.device)



        self.multihead_attn = nn.MultiheadAttention(self.samples_per_subject * self.attention_embedding,
                                                    self.n_heads)#.to(self.device)

        #################################### MHA 2 ###############################################

        # self.key_layer2 = nn.Sequential(
        #     nn.Linear(self.samples_per_subject * self.lstm.output_dim,
        #               self.samples_per_subject * self.attention_embedding),
        # ).to(device)
        #
        # self.value_layer2 = nn.Sequential(
        #     nn.Linear(self.samples_per_subject * self.lstm.output_dim,
        #               self.samples_per_subject * self.attention_embedding),
        # ).to(device)
        #
        # self.query_layer2 = nn.Sequential(
        #     nn.Linear(self.samples_per_subject * self.lstm.output_dim,
        #               self.samples_per_subject * self.attention_embedding),
        # ).to(device)
        #
        # self.multihead_attn2 = nn.MultiheadAttention(self.samples_per_subject * self.attention_embedding,
        #                                             self.n_heads).to(self.device)


        # self.classifier1 = nn.Sequential(
        #     nn.Linear(self.encoder.feature_size * self.n_regions, 1024),  # 115200
        #     # nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(1024, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64)
        #
        # ).to(device)
        self.relu = torch.nn.ReLU()
        self.HS = torch.nn.Hardsigmoid()
        self.HW = torch.nn.Hardswish()
        self.selu = torch.nn.SELU()
        self.celu = torch.nn.CELU()
        self.softplus = torch.nn.Softplus(threshold=20)




#        self.init_weight()
#        self.loadModels()

    def init_weight(self, PT="UFPT"):
        print(self.gain)
        print('init' + PT)
        # return
        if PT == "NPT":
            for name, param in self.query_layer.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')
                # param = param + torch.abs(torch.min(param))
            # for name, param in self.mlp.named_parameters():
            #     if 'weight' in name:
            #         # print(name)
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.mlp2.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            for name, param in self.key_layer.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')
                # param = param + torch.abs(torch.min(param))
            for name, param in self.value_layer.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')
                # param = param + torch.abs(torch.min(param))
            for name, param in self.multihead_attn.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')
                # param = param + torch.abs(torch.min(param))

                # param = param + torch.abs(torch.min(param))

            # for name, param in self.lta_key.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            #
            # for name, param in self.lta_query.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            #
            # for name, param in self.lta_key_components.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            #
            # for name, param in self.lta_query_components.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')

            for name, param in self.gta_embed.named_parameters():
                # print('name = ',name)
                if 'weight' in name:
                    nn.init.kaiming_normal_(param,mode='fan_in')
                # with torch.no_grad():
                #     param.add_(torch.abs(torch.min(param)))
                    # print(torch.min(param))


            for name, param in self.gta_attend.named_parameters():
                # print(name)

                if 'weight' in name:
                    # print(name)
                    # print(param.min())
                    nn.init.kaiming_normal_(param,mode='fan_in')
                # with torch.no_grad():
                #     param.add_(torch.abs(torch.min(param)))

                    # param = param + torch.abs(torch.min(param)) + 1.0

                    # print(name)
                    # print(param.min())
                # param = param + torch.abs(torch.min(param)) + 1.0

                # print(param)
                    # print(torch.min(param))
            # for name, param in self.gta_attend.named_parameters():
            #     print('-------------------')
            #     if 'weight' in name:
            #         print(name)
            #         print(param.min())
            # return

            # for name, param in self.encoder.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.encoder.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.mlp_before_lstm.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.lstm_decoder.named_parameters():
            #     if 'weight' in name:
            #         nn.init.xavier_normal_(param, gain=.35)

            # for name, param in self.query_layer2.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.key_layer2.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.value_layer2.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.multihead_attn2.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            #
            # for name, param in self.attn_time.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            #
            # for name, param in self.attn_weight.named_parameters():
            #     if 'weight' in name:
            #         nn.init.constant_(param, val=0.0001)
            #
            # for name, param in self.attn_spatial.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.attn_region.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.lstm_decoder2.named_parameters():
            #     if 'weight' in name:
            #         nn.init.xavier_normal_(param, gain=.35)
            # for name, param in self.lstm_decoder3.named_parameters():
            #     if 'weight' in name:
            #         nn.init.kaiming_normal_(param,mode='fan_in')
            # for name, param in self.lstm_decoder_time.named_parameters():
            #     if 'weight' in name:
            #         nn.init.xavier_normal_(param, gain=.35)

        # for name, param in self.decoder.named_parameters():
        #     if 'weight' in name:
        #         nn.init.kaiming_normal_(param,mode='fan_in')
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_in')
        # for name, param in self.graph.named_parameters():
        #     if 'weight' in name:
        #         if 'bn1.weight' not in name and 'bn2.weight' not in name and 'bn3.weight' not in name:
        #             nn.init.kaiming_normal_(param,mode='fan_in')


    def loadModels(self):
        if self.PT in ['milc-fMRI', 'variable-attention', 'two-loss-milc']:
            if self.exp in ['UFPT', 'FPT']:
                print('in ufpt and fpt')
                model_dict = torch.load(os.path.join(self.oldpath, 'model' + '.pt'), map_location=self.device)
                self.lstm.load_state_dict(model_dict)
                #self.model.lstm.to(self.device)


    def spatial_attention(self,inputs):

        inputs = inputs.squeeze()
        # print(inputs.shape)
        weights = self.attn_spatial(inputs.reshape(inputs.shape[0] * inputs.shape[1],-1))
        weights = weights.squeeze().reshape(-1, self.n_regions)
        weights = weights.unsqueeze(2).repeat(1, 1, self.time_points)

        inputs = weights * inputs

        return  inputs.unsqueeze(3), weights

    # def means_times_attn_weights(self,means,graph_attention):
    #     return  torch.squeeze(torch.bmm(graph_attention,means.permute(0,2,1)))

    def gta_attention_embeddings(self, x, node_axis=1, dimension='time', mode='train'):
        if dimension == 'time':



            x_readout = x.mean(node_axis, keepdims=True)
            x_readout = (x - x_readout)
            a = x_readout.shape[0]
            b = x_readout.shape[1]
            x_readout = x_readout.reshape(-1, x_readout.shape[2])
            x_embed = self.gta_norm_embeddings(self.gta_embed_embeddings(x_readout))
            x_graphattention = self.gta_attend_embeddings(x_embed).squeeze()
            x_graphattention = (x_graphattention.reshape(a, b))

            return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis)




    # def compute_attn_for_means(self,x,node_axis=1):
    #     # print('x shape = ', x.shape)
    #     x_readout = x.mean(node_axis, keepdim=True)
    #     x_readout = (x * x_readout)
    #     a = x_readout.shape[0]
    #     b = x_readout.shape[1]
    #     x_readout = x_readout.reshape(-1, x_readout.shape[2])
    #     # print('shape = ', x_readout.shape)
    #     x_embed = self.norm(self.embed(x_readout))
    #     x_graphattention = (self.attend(x_embed).squeeze())
    #     x_graphattention = (x_graphattention.reshape(a, b))
    #     x_graphattention = torch.softmax(x_graphattention,dim=1)
    #     values, indices = x_graphattention.max(1)
    #     return torch.unsqueeze(x[np.arange(x.shape[0]),indices,:],dim=1) ,indices

    def gta_attention(self,x,node_axis=1,outputs='', dimension='time',mode='train'):
        if dimension=='time':

            # x_readout = (self.gta_embed(x))
            # a = x_readout.shape[0]
            # b = x_readout.shape[1]
            # x_readout = self.gta_norm(x_readout.reshape(-1, x_readout.shape[2]))
            # x_graphattention = (self.gta_attend(x_readout).squeeze())
            # x_graphattention = torch.sigmoid(x_graphattention.reshape(a, b))
            # print('min = ', torch.min(x))
            x_readout = x.mean(node_axis, keepdim=True)
            x_readout = (x*x_readout)
            a = x_readout.shape[0]
            b = x_readout.shape[1]
            x_readout = x_readout.reshape(-1,x_readout.shape[2])
            x_embed = self.gta_norm(self.gta_embed(x_readout))
            # print('x embed min = ', torch.min(x_embed))
            x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)
            # print('x_graphattention min = ', torch.min(x_graphattention))
            # return
            # print('min = ', x_graphattention.min().item())
            # print('max = ', x_graphattention.max().item())
            x_graphattention = (self.HW(x_graphattention.reshape(a, b)))
            # min = torch.min(x_graphattention,dim=1,keepdim=True).values
            # max = torch.max(x_graphattention,dim=1,keepdim=True).values
            # x_graphattention = (x_graphattention - min) / ((max - min) + 1e-5)


            # x_graphattention = (torch.softmax(x_graphattention,dim=1))

            # v, indexes = torch.kthvalue(x_graphattention, 124, dim=1)
            # v = v.unsqueeze(1).repeat(1, self.time_points)
            # zeros = torch.zeros(v.size()).to(self.device)
            #
            # x_graphattention_new = torch.where((x_graphattention > v), x_graphattention, zeros)

            # print('min = ', x_graphattention.min().item())
            # print('max = ', x_graphattention.max().item())

            # x_graphattention = (x_graphattention - torch.min(x_graphattention,dim=1,keepdim=True).values) / (torch.max(x_graphattention,dim=1,keepdim=True).values - torch.min(x_graphattention,dim=1,keepdim=True).values)
            # x_graphattention = x_graphattention * 2
            # x_graphattention = x_graphattention - 1
            # x = self.gta_norm_x(x)
            # permute_idx = list(range(node_axis)) + [len(x_graphattention.shape) - 1] + list(
            #     range(node_axis, len(x_graphattention.shape) - 1))
            # x_graphattention = x_graphattention.permute(permute_idx)



###############################################################################
            # abs_x_graphattention = torch.abs(x_graphattention)
            # #
            # # if torch.min(x_graphattention[:]) < 0:
            # #     abs_x_graphattention = x_graphattention * -1
            # # else:
            # #     abs_x_graphattention = x_graphattention
            # v, indexes = torch.kthvalue(abs_x_graphattention, 46, dim=1)
            # v = v.unsqueeze(1).repeat(1, self.time_points)
            # zeros = torch.zeros(v.size()).to(self.device)
            # x_graphattention_top = torch.where(abs_x_graphattention > v, x_graphattention, zeros)
            # return (x * (x_graphattention_top.unsqueeze(-1))).mean(node_axis), 'FC2', x_graphattention

            if 'LR' in mode:
                print('in LR')
                # return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis), 'FC2', x_graphattention

                abs_x_graphattention = torch.abs(x_graphattention)
                #
                # if torch.min(x_graphattention[:]) < 0:
                #     abs_x_graphattention = x_graphattention * -1
                # else:
                #     abs_x_graphattention = x_graphattention
                v, indexes = torch.kthvalue(abs_x_graphattention, 147, dim=1)
                v = v.unsqueeze(1).repeat(1, self.time_points)
                zeros = torch.zeros(v.size()).to(self.device)

                x_graphattention_top = torch.where((abs_x_graphattention > v), x_graphattention, zeros)

                v, indexes = torch.kthvalue(abs_x_graphattention, 7, dim=1)
                # print(v)
                v = v.unsqueeze(1).repeat(1, self.time_points)

                zeros = torch.zeros(v.size()).to(self.device)
                # temp = abs_x_graphattention < v
                # sum = torch.sum(temp,dim=1)
                # print(sum)
                x_graphattention_bottom = torch.where((abs_x_graphattention < v), x_graphattention, zeros)

                return (x * (x_graphattention_top.unsqueeze(-1))).mean(node_axis),\
                       (x * (x_graphattention_bottom.unsqueeze(-1))).mean(node_axis), x_graphattention

            return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis), 'FC2', x_graphattention


        else:
            x_q = self.lta_query_components(x.mean(node_axis, keepdims=True))
            x_k = self.lta_key_components(x)
            x_graphattention = (torch.matmul(x_q, x_k.permute(0, 2, 1))).squeeze(1)
            return self.gta_dropout(x * (x_graphattention.unsqueeze(-1))), x_graphattention


    def lta_attention(self,x,node_axis=1, dimension='time',mode='train'):
        if dimension=='time':
            #
            # x = self.relu(x)
            x_q = (self.lta_query(x.mean(node_axis, keepdims=True)))
            x_k = (self.lta_key(x))
            x_graphattention = (torch.matmul(x_q, x_k.permute(0, 2, 1))).squeeze(1)
            # x_graphattention = self.relu(x_graphattention)
            # x = self.relu(x)
            # x_graphattention = torch.softmax(x_graphattention,dim=1)
            # x = torch.abs(x)
            # x_graphattention = self.relu(x_graphattention)


            if 'LR' in mode:
                print('in LR')
                # abs_x_graphattention = torch.abs(x_graphattention)

                if torch.min(x_graphattention[:]) < 0:
                    x_graphattention = x_graphattention * -1
                # else:
                #     abs_x_graphattention = x_graphattention
                v, indexes = torch.kthvalue(x_graphattention, 133, dim=1)
                v = v.unsqueeze(1).repeat(1, self.time_points)
                zeros = torch.zeros(v.size()).to(self.device)
                x_graphattention_top = torch.where(x_graphattention > v, x_graphattention, zeros)

                v, indexes = torch.kthvalue(x_graphattention, 7, dim=1)
                v = v.unsqueeze(1).repeat(1, self.time_points)
                zeros = torch.zeros(v.size()).to(self.device)
                x_graphattention_bottom = torch.where(x_graphattention < v, x_graphattention, zeros)

                return (x * (x_graphattention_top.unsqueeze(-1))).mean(node_axis),\
                       (x * (x_graphattention_bottom.unsqueeze(-1))).mean(node_axis), x_graphattention

            return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis), 'FC2', x_graphattention

        else:
            x_q = self.lta_query_components(x.mean(node_axis, keepdims=True))
            x_k = self.lta_key_components(x)
            x_graphattention = (torch.matmul(x_q, x_k.permute(0, 2, 1))).squeeze(1)
            # print(x_graphattention.shape)
            # v, indexes = torch.kthvalue(x_graphattention, 53, dim=1)
            # v = v.unsqueeze(1).repeat(1, self.n_regions_after)
            # zeros = torch.zeros(v.size()).to(self.device)


            # indices = torch.argsort(x_graphattention,dim=1,descending=False)
            # indices = indices[:,:47]
            #
            # # print(x_graphattention.shape)
            # # print(indices.shape)
            # x_graphattention = torch.gather(x_graphattention,1,indices)
            # indices_cols = indices.unsqueeze(2).repeat(1,1,100)
            # # print(indices_cols.shape)
            # # print(x.shape)
            # x = torch.gather(x, 1, indices_cols)
            # indices_rows = indices.unsqueeze(1).repeat(1,47,1)
            # # print(indices_rows.shape)
            # # print(x.shape)
            # x = torch.gather(x, 2, indices_rows)
            # print(x.shape)
            # print(x_graphattention.shape)


            # print(x_graphattention.shape)


            # x_graphattention = x_graphattention[x_graphattention > v]
            # print(x_graphattention.shape)
            # return
            # x_graphattention = torch.where(x_graphattention > v, x_graphattention, zeros)
            return (x * (x_graphattention.unsqueeze(-1))), x_graphattention

        # v, indexes = torch.kthvalue(x_graphattention, 46, dim=1)
        # v = v.unsqueeze(1).repeat(1, self.time_points)
        # zeros = torch.zeros(v.size()).to(self.device)
        # x_graphattention = torch.where(x_graphattention < v, x_graphattention, zeros)

    def get_attention(self, outputs,type="weight"):

        if type=="weight":
            weights = self.attn_weight(outputs.reshape(outputs.shape[0] * outputs.shape[1],-1))
            weights = weights.squeeze().reshape(-1,self.time_points)
            # v, indexes = torch.kthvalue(weights, 46, dim=1)
            # v = v.unsqueeze(1).repeat(1, self.time_points)
            # zeros = torch.zeros(v.size()).to(self.device)
            # weights = torch.where(weights < v, weights, zeros)
            # sample = torch.rand(weights.shape[0], weights.shape[1]).topk(124, dim=1).indices.to(self.device)
            # weights = weights.scatter_(dim=1, index=sample, value=0.0)
            # weights[mask]

        elif type=="region":
            weights = self.attn_region(outputs.reshape(outputs.shape[0] * outputs.shape[1],-1))
            weights = weights.squeeze().reshape(-1,self.n_regions)
        elif type=="time":
            weights = self.attn_time(outputs.reshape(outputs.shape[0] * outputs.shape[1],-1))
            weights = weights.squeeze().reshape(-1,self.time_points)



        normalized_weights = weights #torch.softmax(weights,dim=1)

        if type == "weight":
            # print('type')
            # sample = torch.rand(normalized_weights.shape[0], normalized_weights.shape[1]).topk(124, dim=1).indices.to(self.device)
            # normalized_weights = normalized_weights.scatter_(dim=1, index=sample, value=0.0)
            v, indexes = torch.kthvalue(normalized_weights, 46, dim=1)
            v = v.unsqueeze(1).repeat(1, self.time_points)
            zeros = torch.zeros(v.size()).to(self.device)
            normalized_weights = torch.where(normalized_weights < v, normalized_weights, zeros)

        # normalized_weights -= normalized_weights.min(1, keepdim=True)[0]
        # normalized_weights /= normalized_weights.max(1, keepdim=True)[0]

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        # attn_applied = normalized_weights * outputs

        attn_applied = attn_applied.squeeze()
        # logits = self.decoder(attn_applied)
        #print("attention decoder ", time.time() - t)'
        # if type == 'time' or type == 'region':
        return attn_applied, normalized_weights.unsqueeze(1)
        # else:
        #     return attn_applied

    def multi_head_attention(self, outputs, k, FNC="", FNC2=""):



        key = self.key_layer(outputs)
        value = self.value_layer(outputs)
        query = self.query_layer(outputs)
        key = key.permute(1,0,2)
        value = value.permute(1, 0, 2)
        query = query.permute(1, 0, 2)


        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        attn_output = attn_output.permute(1,0,2)

        attn_output_weights = attn_output_weights #+ FNC + FNC2
        return attn_output, attn_output_weights

    def multi_head_attention2(self, outputs, k, FNC="", FNC2=""):



        key = self.key_layer2(outputs)
        value = self.value_layer2(outputs)
        query = self.query_layer2(outputs)
        key = key.permute(1,0,2)
        value = value.permute(1, 0, 2)
        query = query.permute(1, 0, 2)


        attn_output, attn_output_weights = self.multihead_attn2(query, key, value)
        attn_output = attn_output.permute(1,0,2)

        attn_output_weights = attn_output_weights #+ FNC + FNC2
        return attn_output, attn_output_weights


    def get_topk_weights(self,weights):

        weights = weights.reshape(weights.shape[0],-1)
        sorted_weights = torch.argsort(weights, descending=True)
        top_k_weights = sorted_weights[:,:self.k]
        weights = weights.gather(1, sorted_weights)
        weights = weights[:,:self.k]
        return weights.reshape(weights.shape[0],-1), \
               top_k_weights




    def get_lstm_loss(self, data, B):
        encoder_logits = self.lstm_decoder(data.permute(0,1,2,3).reshape(B,-1))
        return encoder_logits

    def final_attention(self, data):
        weights = self.mlp2(data)
        data = weights * data

        return data, weights


    def create_graphs(self, dataset, device):
        data_list = []
        # self.data_tensor = self.data_tensor.reshape(self.data_tensor.shape[0] * self.data_tensor.shape[1], -1).contiguous()

        # process by session_id
        # grouped = df.groupby('session_id')
        for subject, label, edge_weights in dataset:
            n_nodes = subject.shape[0]
            node_features = subject
            edge_index = torch.from_numpy(np.arange(0, n_nodes)).int()
            # a = (edge_index.unsqueeze(0).repeat(n_nodes, 1))
            # b = (edge_index.unsqueeze(1).repeat(1, n_nodes))
            edge_index = torch.cat((edge_index.unsqueeze(1).repeat(1, n_nodes).reshape(1, -1),
                                    edge_index.unsqueeze(0).repeat(n_nodes, 1).reshape(1, -1)), dim=0).long()
            edge_index = edge_index.to(device)
            # print(edge_index)
            # return
            x = node_features

            y = label.float().view(1)

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weights)
            data_list.append(data)

        return data_list


    def forward(self, input, targets, mode='train', device="cpu",epoch=0, FNC = ""):
        indices = ""
        # r = random.randint(0, 800)
        # sx = sx[:,r:r+400,:,:]

        B = input.shape[0]
        W = input.shape[1]
        R = input.shape[2]
        T = input.shape[3]
        input = input.reshape(B,1,self.time_points,R,T)
        input = input.permute(1,0,2,3,4)
        (FC_logits), FC,  FC_sum, FC_time_weights = 0., 0., 0., 0.

        for sb in range(1):

            sx = input[sb,:,:,:,:]
            B = sx.shape[0]
            W = sx.shape[1]
            R = sx.shape[2]
            T = sx.shape[3]
            # inputs = self.mlp_before_lstm(sx.squeeze().reshape(B*W,R))
            # R = self.n_regions_after
            # inputs = inputs.reshape(B,W,R,1).contiguous()



            inputs = sx.permute(0, 2, 1, 3).contiguous()


            # print(inputs.shape)
            # inputs, spatial_weights = self.spatial_attention(inputs)
            # print(inputs.shape)
            inputs = inputs.reshape(B*R,W,T)
            # inputs = inputs.unsqueeze(1)
            #
            #
            # # print(inputs.shape)
            # # return
            # inputs = self.encoder(inputs).to(self.device)
            # W = inputs.shape[1]
            # T = inputs.shape[2]
            # inputs = inputs.reshape(B,R,W,T)
            inputs = (self.lstm(inputs))
            # inputs = inputs.to(self.device)


            inputs = inputs.reshape(B,R,W,inputs.shape[2])


            # lstm_output_before_MHA = self.lstm_decoder(torch.sum(inputs,dim=2).squeeze().reshape(B, -1))
            # print(inputs.shape)
            inputs = inputs.permute(2,0,1,3).contiguous()
            # print(inputs.shape)
            inputs = inputs.reshape(W*B,R,self.lstm.output_dim)

            # features = []
            # weights = []
            # for time_point in inputs:
            #     f,w = self.multi_head_attention(time_point, self.k)
            #     features.append(f)
            #     weights.append(w)
            # outputs = torch.stack(features)
            # attn_weights = torch.stack(weights)
            # inputs=inputs.to(self.device)
            outputs , attn_weights = self.multi_head_attention(inputs,self.k)
            # print(self.device_two)

            # attn_weights = attn_weights.to(self.device_two)
            # outputs = outputs.to(self.device_two)

            attn_weights = attn_weights.reshape(W,B,R,R)
            # attn_weights = attn_weights.to(self.device_two)
            # print("#############")

            # print(attn_weights.shape)

            # print(inputs.shape)

            # return
            # lstm_output_after_MHA = self.lstm_decoder_time(torch.sum(outputs, dim=2).squeeze().reshape(B,-1))

            attn_weights = attn_weights.permute(1, 0, 2, 3).contiguous()

            ################################# Code for states and means START##############################
            # means = self.auto_encoder(attn_weights.reshape(B*W,1,R,R))
            # # print('means shape = ',means.shape)
            # selected_means, selected_indices = self.compute_attn_for_means(means)
            # # means_weights = means_weights.reshape(B,W,-1)
            # # print('selected means shape = ',selected_means.shape)
            # # print('selected indices shape = ', selected_indices.shape)
            #
            # ENC_from_means = self.auto_decoder(selected_means.reshape(B*W,1,8,8))
            # # print('ENC shape = ',ENC_from_means.shape)
            # means = self.means_to_higher_projection(means)
            # # means = means.reshape(B,W,self.n_clusters,self.n_regions_after,self.n_regions_after)
            # means_logits = self.means_times_attn_weights(means,attn_weights.reshape(B*W,1,R*R))
            # # print('means logits shape = ',means_logits.shape)

            ################################# Code for states and means END##############################
            # return
            attn_weights = attn_weights.reshape(B, W, -1)
            # inputs = inputs.permute(1, 0, 2, 3).contiguous()

            # inputs = inputs.reshape(B*W,R,inputs.shape[3])
            # inputs, attention_regions_weights = self.get_attention(inputs,type="region")
            # attention_regions_weights = attention_regions_weights.reshape(B,W,R).permute(0,2,1)
            # print(attention_regions_weights.shape)


            # inputs = inputs.reshape(B,W,self.lstm.output_dim)


            # outputs,attn_weights_time = self.multi_head_attention2(inputs,self.k)
            # inputs, attention_time_weights = self.get_attention(inputs,type="time")
            # if mode == 'eval':
            #     print(torch.min(attention_time_weights[:]))
            #     print(torch.max(attention_time_weights[:]))
            #
            #     print(torch.min(attention_regions_weights[:]))
            #     print(torch.max(attention_regions_weights[:]))

            # print(attention_time_weights.shape)
            # return
            # FC, FC_time_weights = self.get_attention(attn_weights,type='weight')
            # print(attn_weights.shape)

            # outputs = outputs.reshape(W, B, R, self.attention_embedding).permute(1, 0, 2, 3).contiguous()
            # outputs = outputs.reshape(B,W,-1)

            FC, FC2, FC_time_weights = self.gta_attention(attn_weights,dimension='time',mode=mode)
            # FC = FC.to(self.device_two)
            # FC2 = FC2.to(self.device_two)
            # FC_time_weights = FC_time_weights.to(self.device_two)
            # product = torch.bmm(attn_weights,attn_weights.permute(0,2,1))
            # reg_ortho = (product / torch.amax(product,dim=(1,2)).unsqueeze(-1).unsqueeze(-1) - torch.eye(W, device=self.device))\
            #     .triu().norm(dim=(1, 2)).mean()
            # print(FC.shape)
            # print(FC_time_weights.shape)
            # return

            FC = FC.squeeze().reshape(B,R,R)
            # outputs_emb = outputs_emb.squeeze().reshape(B, R, self.attention_embedding)
            # FC = FC.view(FC.size(0), -1)
            # FC = torch.abs(FC)
            # FC -= FC.min(1, keepdim=True)[0]
            # FC /= FC.max(1, keepdim=True)[0]
            # lt = FC[:]>0.7
            # FC[lt]=1
            # FC = FC.view(B, R, R)



            # print(FC.shape)
            # for i in range(B):
            #     sub = FC[i, :, :]
            #     FC[i, :, :] = (sub - sub.min()) / (sub.max() - sub.min())
            # lt = FC[:]<0.5
            # FC[lt]=0
            # print(FC.shape)
            # FC, FC_components_weights = self.final_attention(FC.reshape(-1,1))#self.lta_attention(FC, dimension='components',mode=mode)
            # FC = FC.reshape(B, R, R)
            # print(FC.shape)
            # return
            # FC = torch.bmm(attention_time_weights,attn_weights).squeeze().reshape(B,self.n_regions,self.n_regions)
            FC_sum =  torch.mean(attn_weights,dim=1).squeeze().reshape(B,R,R)

            # FC = FC.to(self.device)

            # FC_logits = self.mlp((FC.reshape(B,-1)))
            if sb ==0:
                FC_logits = self.encoder((FC.unsqueeze(1)))
            else:
                FC_logits += self.encoder((FC.unsqueeze(1)))
            # outputs_logits = self.mlp2(outputs_emb.reshape(B, -1))
    ###############################################################################
            # print('starting graph')

            # outputs = outputs.reshape(W, B, R, self.attention_embedding).permute(1, 0, 2, 3).contiguous()
            # outputs = outputs.reshape(B,W,-1)
            # outputs,_,_ = self.gta_attention_embeddings(outputs,dimension='time',mode=mode)
            # outputs = outputs.reshape(B, R, self.attention_embedding)
            # outputs = outputs.to(self.device_two)
            # FC = torch.transpose(FC,1,2)
            # FC = FC.to(self.device_two)
            # # print('outputs shape = ',outputs.shape)
            # # print('FNC shape = ', FC.shape)
            # outputs = TensorDataset(outputs, targets, FC.reshape(B,-1))
            # # print('Tensor data')
            # outputs = self.create_graphs(outputs, self.device_two)
            # # print('create graphs')
            # # print("graph creation time", time.time() - t)
            # #
            # # t = time.time()
            # # inputs = inputs.shuffle()
            # outputs = DataLoader(outputs, batch_size=B)
            # # print('data loader')
            # graph_logits = ''
            # for data in outputs:
            #     data.batch = data.batch.to(self.device_two)
            #     # print(data.batch.device)
            #     # print(data.x.device)
            #     # print(data.y.device)
            #     # print(data.edge_index.device)
            #     # print(data.edge_attr.device)
            #     outputs,region_indices = self.graph(data)

    #############################################################################

            # inputs = torch.sum(inputs,dim=2).squeeze()
            # inputs = torch.sum(inputs,dim=1).squeeze()
            # inputs =  inputs  + lstm_output_after_MHA #+ lstm_output_before_MHA
            # logits = self.decoder(inputs)

            # print(attn_weights.shape)
            # attn_weights = attn_weights.reshape(B,W,-1)
            # final_weights = torch.sum(attn_weights,dim=1).squeeze()
            # print(attn_weights.shape)

            # final_weights = self.get_attention(attn_weights)
            # print(final_weights.shape)
            # lstm_logits = self.lstm_decoder(inputs.reshape(B, -1))

            # logits = self.mlp(final_weights)
            # if mode=='test':
            # FC_time_weights = torch.rand(B,self.time_points)
            # FC_logits = FC_logits.to(self.device_two)
        if mode == 'test':
            return (FC_logits) , FC, FC2, FC_sum, FC_time_weights.squeeze(), attn_weights#, means_logits,selected_indices,ENC_from_means

        return (FC_logits) , FC, FC2, FC_sum, FC_time_weights.squeeze(), 'attn_weights'#, means_logits,selected_indices,ENC_from_means
        # else:
        #     return logits + self.lstm_decoder2(lstm_output_before_MHA)  + self.lstm_decoder3(lstm_output_after_MHA), 'FC'


