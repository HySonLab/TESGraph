#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Some codes here are modified from SuperGluePretrainedNetwork https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
#
import math
import importlib
import torch
from .network_util import build_mlp, Gen_Index, Aggre_Index, MLP
from .networks_base import BaseNetwork
import inspect
from collections import OrderedDict
import os
from codeLib.utils import onnx
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
import torch.nn as nn
from typing import Optional
from copy import deepcopy
from torch_scatter import scatter
from codeLib.common import filter_args_create
import ssg
from .EGCL import E_GCL
from .Temporal_graph import EvolveGCNO 




class TripletGCN(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='mean', with_bn=True):
        super().__init__(aggr=aggr)
        # print('============================')
        # print('aggr:',aggr)
        # print('============================')
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden
        self.nn1 = build_mlp([dim_node*2+dim_edge, dim_hidden, dim_hidden*2+dim_edge],
                             do_bn=with_bn, on_last=True)
        self.nn2 = build_mlp([dim_hidden, dim_hidden, dim_node], do_bn=with_bn)

        self.reset_parameter()

    def reset_parameter(self):
        pass
        # reset_parameters_with_activation(self.nn1[0], 'relu')
        # reset_parameters_with_activation(self.nn1[3], 'relu')
        # reset_parameters_with_activation(self.nn2[0], 'relu')

    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(
            edge_index, x=x, edge_feature=edge_feature)
        gcn_x = x + self.nn2(gcn_x)
        return gcn_x, gcn_e

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat([x_i, edge_feature, x_j], dim=1)
        x = self.nn1(x)  # .view(b,-1)
        new_x_i = x[:, :self.dim_hidden]
        new_e = x[:, self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        new_x_j = x[:, (self.dim_hidden+self.dim_edge):]
        x = new_x_i+new_x_j
        return [x, new_e]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)
        return x


class TripletGCNModel(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(TripletGCN(**kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(
                node_feature, edge_feature, edges_indices)

            if i < (self.num_layers-1):
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
        return node_feature, edge_feature


class MessagePassing_IMP(MessagePassing):
    def __init__(self, dim_node, aggr='mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        # Attention layer
        self.subj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.subj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_feature, edge_index):
        node_msg, edge_msg = self.propagate(
            edge_index, x=x, edge_feature=edge_feature)
        return node_msg, edge_msg

    def message(self, x_i, x_j, edge_feature):
        '''Node'''
        message_pred_to_subj = self.subj_node_gate(
            torch.cat([x_i, edge_feature], dim=1)) * edge_feature  # n_rel x d
        message_pred_to_obj = self.obj_node_gate(
            torch.cat([x_j, edge_feature], dim=1)) * edge_feature  # n_rel x d
        node_message = (message_pred_to_subj+message_pred_to_obj)

        '''Edge'''
        message_subj_to_pred = self.subj_edge_gate(
            torch.cat([x_i, edge_feature], 1)) * x_i  # nrel x d
        message_obj_to_pred = self.obj_edge_gate(
            torch.cat([x_j, edge_feature], 1)) * x_j  # nrel x d
        edge_message = (message_subj_to_pred+message_obj_to_pred)

        return [node_message, edge_message]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)
        return x


class MessagePassing_VGfM(MessagePassing):
    def __init__(self, dim_node, aggr='mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.subj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.subj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.geo_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_feature, geo_feature, edge_index):
        node_msg, edge_msg = self.propagate(
            edge_index, x=x, edge_feature=edge_feature, geo_feature=geo_feature)
        return node_msg, edge_msg

    def message(self, x_i, x_j, edge_feature, geo_feature):
        message_pred_to_subj = self.subj_node_gate(
            torch.cat([x_i, edge_feature], dim=1)) * edge_feature  # n_rel x d
        message_pred_to_obj = self.obj_node_gate(
            torch.cat([x_j, edge_feature], dim=1)) * edge_feature  # n_rel x d
        node_message = (message_pred_to_subj+message_pred_to_obj)

        message_subj_to_pred = self.subj_edge_gate(
            torch.cat([x_i, edge_feature], 1)) * x_i  # nrel x d
        message_obj_to_pred = self.obj_edge_gate(
            torch.cat([x_j, edge_feature], 1)) * x_j  # nrel x d
        message_geo = self.geo_edge_gate(
            torch.cat([geo_feature, edge_feature], 1)) * geo_feature
        edge_message = (message_subj_to_pred+message_obj_to_pred+message_geo)

        # x = torch.cat([x_i,edge_feature,x_j],dim=1)
        # x = self.nn1(x)#.view(b,-1)
        # new_x_i = x[:,:self.dim_hidden]
        # new_e   = x[:,self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        # new_x_j = x[:,(self.dim_hidden+self.dim_edge):]
        # x = new_x_i+new_x_j
        return [node_message, edge_message]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)
        return x


class MessagePassing_Gate(MessagePassing):
    def __init__(self, dim_node, aggr='mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.temporal_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        x_i = self.temporal_gate(torch.cat([x_i, x_j], dim=1)) * x_i
        return x_i


class TripletIMP(torch.nn.Module):
    def __init__(self, dim_node, num_layers, aggr='mean', **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dim_node = dim_node
        self.edge_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)
        self.node_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)
        self.msp_IMP = MessagePassing_IMP(dim_node=dim_node, aggr=aggr)
        self.reset_parameter()

    def reset_parameter(self):
        pass

    def forward(self, data):
        '''shortcut'''
        x = data['roi'].x
        edge_feature = data['edge2D'].x
        edge_index = data['roi', 'to', 'roi'].edge_index

        '''process'''
        x = self.node_gru(x)
        edge_feature = self.edge_gru(edge_feature)
        for i in range(self.num_layers):
            node_msg, edge_msg = self.msp_IMP(
                x=x, edge_feature=edge_feature, edge_index=edge_index)
            x = self.node_gru(node_msg, x)
            edge_feature = self.edge_gru(edge_msg, edge_feature)
        return x, edge_feature


class TripletVGfM(torch.nn.Module):
    def __init__(self, dim_node, num_layers, aggr='mean', **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dim_node = dim_node
        self.edge_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)
        self.node_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)

        self.msg_vgfm = MessagePassing_VGfM(dim_node=dim_node, aggr=aggr)
        self.msg_t_node = MessagePassing_Gate(dim_node=dim_node, aggr=aggr)
        self.msg_t_edge = MessagePassing_Gate(dim_node=dim_node, aggr=aggr)

        self.edge_encoder = ssg.models.edge_encoder.EdgeEncoder_VGfM()

        self.reset_parameter()

    def reset_parameter(self):
        pass
        # reset_parameters_with_activation(self.nn1[0], 'relu')
        # reset_parameters_with_activation(self.nn1[3], 'relu')
        # reset_parameters_with_activation(self.nn2[0], 'relu')

    def forward(self, data):
        '''shortcut'''
        x = data['roi'].x
        edge_feature = data['edge2D'].x
        edge_index = data['roi', 'to', 'roi'].edge_index
        geo_feature = data['roi'].desp
        temporal_node_graph = data['roi', 'temporal', 'roi'].edge_index
        temporal_edge_graph = data['edge2D', 'temporal', 'edge2D'].edge_index

        '''process'''
        x = self.node_gru(x)
        edge_feature = self.edge_gru(edge_feature)
        extended_geo_feature = self.edge_encoder(geo_feature, edge_index)
        for i in range(self.num_layers):
            node_msg, edge_msg = self.msg_vgfm(
                x=x, edge_feature=edge_feature, geo_feature=extended_geo_feature, edge_index=edge_index)
            if temporal_node_graph.shape[0] == 2:
                temporal_node_msg = self.msg_t_node(
                    x=x, edge_index=temporal_node_graph)
                node_msg += temporal_node_msg
            if temporal_edge_graph.shape[0] == 2:
                temporal_edge_msg = self.msg_t_edge(
                    x=edge_feature, edge_index=temporal_edge_graph)
                edge_msg += temporal_edge_msg
            x = self.node_gru(node_msg, x)
            edge_feature = self.edge_gru(edge_msg, edge_feature)

        return x, edge_feature


class MSG_MV_DIRECT(MessagePassing):
    def __init__(self, aggr: str, use_res: bool = True):
        super().__init__(aggr=aggr,
                         flow='source_to_target')
        self.use_res = use_res

    def forward(self, node, images, edge_index):
        dummpy = (images, node)
        return self.propagate(edge_index, x=dummpy, node=node)

    def message(self, x_j):
        """

        Args:
            x_j (_type_): image_feature
        """
        return x_j

    def update(self, x, node):
        if self.use_res:
            x += node
        return x


class MSG_FAN(MessagePassing):
    def __init__(self,
                 dim_node: int, dim_edge: int, dim_atten: int,
                 num_heads: int,
                 use_bn: bool,
                 aggr='sum',
                 attn_dropout: float = 0.5,
                 flow: str = 'target_to_source'):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)

        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj, self.dim_node_proj+self.dim_edge_proj,
                           self.dim_edge_proj])

        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_k = build_mlp([dim_edge, dim_edge])
        self.proj_v = build_mlp([dim_node, dim_atten])

        self.nn_edge = build_mlp([dim_node*2+dim_edge, (dim_node+dim_edge), dim_edge],
                                 do_bn=use_bn, on_last=False)

        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()

        '''update'''
        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                     do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index):
        return self.propagate(edge_index, x=x, edge_feature=edge_feature, x_ori=x)

    def message(self, x_i: Tensor, x_j: Tensor, edge_feature: Tensor) -> Tensor:
        '''
        x_i [N, D_N]
        x_j [N, D_N]
        '''
        num_node = x_i.size(0)

        '''triplet'''
        triplet_feature = torch.cat([x_i, edge_feature, x_j], dim=1)
        triplet_feature = self.nn_edge(triplet_feature)

        '''FAN'''
        # proj
        x_i = self.proj_q(x_i).view(
            num_node, self.dim_node_proj, self.num_head)  # [N,D,H]
        edge = self.proj_k(edge_feature).view(
            num_node, self.dim_edge_proj, self.num_head)  # [M,D,H]
        x_j = self.proj_v(x_j)
        # est attention
        att = self.nn_att(torch.cat([x_i, edge], dim=1))  # N, D, H
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        value = prob.reshape_as(x_j)*x_j

        return [value, triplet_feature, prob]

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        inputs[0] = scatter(inputs[0], index, dim=self.node_dim,
                            dim_size=dim_size, reduce=self.aggr)
        return inputs

    def update(self, x, x_ori):
        x[0] = self.update_node(torch.cat([x_ori, x[0]], dim=1))
        return x


class JointGNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.with_geo = kwargs['with_geo']
        self.num_layers = kwargs['num_layers']
        self.num_heads = kwargs['num_heads']
        dim_node = kwargs['dim_node']
        dim_edge = kwargs['dim_edge']
        drop_out_p = kwargs['drop_out']
        self.gconvs = torch.nn.ModuleList()

        # Get version
        args_jointgnn = kwargs['jointgnn']
        args_img_msg = kwargs[args_jointgnn['img_msg_method']]

        gnn_modules = importlib.import_module(
            'ssg.models.network_GNN').__dict__
        # jointGNNModel = gnn_modules['JointGNN_{}'.format(args_jointgnn['version'].lower())]
        img_model = gnn_modules[args_jointgnn['img_msg_method']]
        self.msg_img = filter_args_create(
            img_model, {**kwargs, **args_img_msg})

        # GRU
        self.node_gru = nn.GRUCell(input_size=dim_node, hidden_size=dim_node)
        self.edge_gru = nn.GRUCell(input_size=dim_edge, hidden_size=dim_edge)

        # gate
        if self.with_geo:
            self.geo_gate = nn.Sequential(
                nn.Linear(dim_node * 2, 1), nn.Sigmoid())

        self.drop_out = None
        if drop_out_p > 0:
            self.drop_out = torch.nn.Dropout(drop_out_p)

        # for _ in range(self.num_layers):
        #     self.gconvs.append(jointGNNModel(**kwargs))

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(
                MSG_FAN, {**kwargs, **kwargs['MSG_FAN']}))

    def forward(self, data):
        probs = list()
        node = data['node'].x
        if self.with_geo:
            geo_feature = data['geo_feature'].x
        # image = data['roi'].x
        edge = data['node', 'to', 'node'].x
        # spatial = data['node'].spatial if 'spatial' in data['node'] else None
        edge_index_node_2_node = data['node', 'to', 'node'].edge_index
        # edge_index_image_2_ndoe = data['roi','sees','node'].edge_index

        # TODO: use GRU?
        node = self.node_gru(node)
        edge = self.edge_gru(edge)
        for i in range(self.num_layers):
            gconv = self.gconvs[i]

            if self.with_geo:
                geo_msg = self.geo_gate(torch.cat(
                    (node, geo_feature), dim=1)) * torch.sigmoid(geo_feature)  # TODO:put the gate back
                # geo_msg = self.geo_gate(torch.cat((node,geo_feature),dim=1)) * geo_feature
                node += geo_msg

            # node, edge, prob = gconv(node,image,edge,edge_index_node_2_node,edge_index_image_2_ndoe)
            node_msg, edge_msg, prob = gconv(
                node, edge, edge_index_node_2_node)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_msg = torch.nn.functional.relu(node_msg)
                edge_msg = torch.nn.functional.relu(edge_msg)

                if self.drop_out:
                    node_msg = self.drop_out(node_msg)
                    edge_msg = self.drop_out(edge_msg)

            node = self.node_gru(node_msg, node)
            edge = self.edge_gru(edge_msg, edge)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node, edge, probs

class JointEGNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(JointEGNN, self).__init__()
        self.with_geo = kwargs['with_geo']
        self.num_layers = kwargs['num_layers']
        self.num_heads = kwargs['num_heads']
        dim_node = kwargs['dim_node']
        dim_edge = kwargs['dim_edge']
        drop_out_p = kwargs['drop_out']
        self.gconvs = torch.nn.ModuleList()

        # Get version
        args_jointgnn = kwargs['jointgnn']
        args_img_msg = kwargs[args_jointgnn['img_msg_method']]

        gnn_modules = importlib.import_module(
            'ssg.models.network_GNN').__dict__
        # jointGNNModel = gnn_modules['JointGNN_{}'.format(args_jointgnn['version'].lower())]
        img_model = gnn_modules[args_jointgnn['img_msg_method']]
        self.msg_img = filter_args_create(
            img_model, {**kwargs, **args_img_msg})

        # GRU
        self.node_gru = nn.GRUCell(input_size=dim_node, hidden_size=dim_node)
        self.edge_gru = nn.GRUCell(input_size=dim_edge, hidden_size=dim_edge)

        # gate
        if self.with_geo:
            self.geo_gate = nn.Sequential(
                nn.Linear(dim_node * 2, 1), nn.Sigmoid())

        self.drop_out = None
        if drop_out_p > 0:
            self.drop_out = torch.nn.Dropout(drop_out_p)

        # for _ in range(self.num_layers):
        #     self.gconvs.append(jointGNNModel(**kwargs))

        for i in range(self.num_layers):
            self.gconvs.append(filter_args_create(
                MSG_FAN, {**kwargs, **kwargs['MSG_FAN']}))
            
            self.add_module("gcl_%d" % i, E_GCL(dim_node, dim_node, dim_node, edges_in_d=dim_edge,
                                    act_fn=nn.SiLU(), residual=True, attention=True,
                                    normalize=False, tanh=False))
            
            #self.add_module("fc_%d" % i, nn.Linear(dim_node, dim_edge))
        self.edge_out = nn.Linear(dim_node, dim_edge)
        #self.node_in = nn.Linear(dim_node, dim_edge)
        #self.node_out = nn.Linear(dim_edge, dim_node)

    def forward(self, data):
        probs = list()
        h = data['node'].x #node
        if self.with_geo:
            geo_feature = data['geo_feature'].x
        # image = data['roi'].x
        edge_attr = data['node', 'to', 'node'].x #edge
        edges = data['node', 'to', 'node'].edge_index #edge_index_node_2_node

        # TODO: use GRU?
        h = self.node_gru(h)
        edge_attr = self.edge_gru(edge_attr)
        for i in range(self.num_layers):
            gconv = self.gconvs[i]

            if self.with_geo:
                geo_msg = self.geo_gate(torch.cat(
                    (h, geo_feature), dim=1)) * torch.sigmoid(geo_feature)  # TODO:put the gate back
                h += geo_msg

            node_msg, edge_msg, prob = gconv(
                h, edge_attr, edges)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_msg = torch.nn.functional.relu(node_msg)
                edge_msg = torch.nn.functional.relu(edge_msg)

                if self.drop_out:
                    node_msg = self.drop_out(node_msg)
                    edge_msg = self.drop_out(edge_msg)
            
            #h = self.node_in(h)
            node_msg, geo_feature, _, edge_msg = self._modules["gcl_%d" % i](node_msg, edges, geo_feature, edge_attr=edge_msg)
            #h = self.node_out(h)
            edge_msg = self.edge_out(edge_msg)
            h = self.node_gru(node_msg, h)
            edge_attr = self.edge_gru(edge_msg, edge_attr)

        return h, edge_attr

class EGNN(nn.Module):
    # def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
    def __init__(self, **kwargs):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        #{'dim_node': 256, 'dim_edge': 256, 'dim_atten': 256, 'num_layers': 2, 'num_heads': 8, 
        #'aggr': 'max', 'DROP_OUT_ATTEN': 0.3, 'use_bn': False}


        # self.hidden_nf = hidden_nf
        # self.device = device
        # self.n_layers = n_layers
        # self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        # self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        # for i in range(0, n_layers):
        #     self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
        #                                         act_fn=act_fn, residual=residual, attention=attention,
        #                                         normalize=normalize, tanh=tanh))
        # self.to(self.device)

        #-------------------------------------------------
        self.num_layers = kwargs['num_layers']
        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        self.hidden_nf = kwargs['dim_node']

        self.embedding_in = nn.Linear(kwargs['dim_node'], self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, kwargs['dim_node'])
            

        for i in range(self.num_layers):
            self.gconvs.append(filter_args_create(MSG_FAN, kwargs))
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=kwargs['dim_edge'],
                                                act_fn=nn.SiLU(), residual=True, attention=True,
                                                normalize=False, tanh=False))

        # self.add_module("Temporal Layer", EvolveGCNO(
        #                                             in_channels = self.hidden_nf,
        #                                             improved= False,
        #                                             cached= False,
        #                                             normalize= True,
        #                                             add_self_loops= True))
    # def forward(self, h, x, edges, edge_attr):
    #     h = self.embedding_in(h)
    #     for i in range(0, self.n_layers):
    #         h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
    #     h = self.embedding_out(h)
    #     return h, x
        
    def forward(self, data):
        h = data['node'].x[:]
        edges = data['node', 'to', 'node'].edge_index
        edge_attr = data['node', 'to', 'node'].x
        x = data['coord'].x
        h = self.embedding_in(h)
        for i in range(0, self.num_layers):
            gconv = self.gconvs[i]
            h, edge_attr, prob = gconv(h, edge_attr, edges)

            if i < (self.num_layers-1) or self.num_layers == 1:
                h = torch.nn.functional.relu(h)
                edge_attr = torch.nn.functional.relu(edge_attr)

                if self.drop_out:
                    h = self.drop_out(h)
                    edge_attr = self.drop_out(edge_attr)

            #if prob is not None:
                #probs.append(prob.cpu().detach())
            #else:
                #probs.append(None)
            h, x, _, edge_feat = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
            edge_attr = edge_feat

        h = self.embedding_out(h)

        return h, x, edge_attr
    
class GraphEdgeAttenNetworkLayers(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(MSG_FAN, kwargs))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, prob = gconv(
                node_feature, edge_feature, edges_indices)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs


class FAN_GRU(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        dim_node = kwargs['dim_node']
        dim_edge = kwargs['dim_edge']
        self.node_gru = nn.GRUCell(input_size=dim_node, hidden_size=dim_node)
        self.edge_gru = nn.GRUCell(input_size=dim_edge, hidden_size=dim_edge)

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(MSG_FAN, kwargs))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index

        # Init GRU
        node_feature = self.node_gru(node_feature)
        edge_feature = self.edge_gru(edge_feature)

        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_msg, edge_msg, prob = gconv(
                node_feature, edge_feature, edges_indices)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_msg = torch.nn.functional.relu(node_msg)
                edge_msg = torch.nn.functional.relu(edge_msg)

                if self.drop_out:
                    node_msg = self.drop_out(node_msg)
                    edge_msg = self.drop_out(edge_msg)

            node_feature = self.node_gru(node_msg, node_feature)
            edge_feature = self.edge_gru(edge_msg, edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs


class FAN_GRU_2(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        dim_node = kwargs['dim_node']
        dim_edge = kwargs['dim_edge']
        self.node_gru = nn.GRUCell(input_size=dim_node, hidden_size=dim_node)
        self.edge_gru = nn.GRUCell(input_size=dim_edge, hidden_size=dim_edge)

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(MSG_FAN, kwargs))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index

        # Init GRU
        node_feature = self.node_gru(node_feature)
        edge_feature = self.edge_gru(edge_feature)

        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_msg, edge_msg, prob = gconv(
                node_feature, edge_feature, edges_indices)

            # if i < (self.num_layers-1) or self.num_layers==1:
            #     node_msg = torch.nn.functional.relu(node_msg)
            #     edge_msg = torch.nn.functional.relu(edge_msg)

            #     if self.drop_out:
            #         node_msg = self.drop_out(node_msg)
            #         edge_msg = self.drop_out(edge_msg)

            node_feature = self.node_gru(node_msg, node_feature)
            edge_feature = self.edge_gru(edge_msg, edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs
