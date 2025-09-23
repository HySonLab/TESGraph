#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:44:20 2021

@author: sc
"""
from .trainer_SGFN import Trainer_SGFN
from .trainer_IMP import Trainer_IMP
from .trainer_ESGNN import Trainer_ESGNN
trainer_dict = {
    'sgfn': Trainer_SGFN,
    'sgpn': Trainer_SGFN,
    'imp': Trainer_IMP,
    'jointsg': Trainer_SGFN,
    'esgnn': Trainer_ESGNN
}
