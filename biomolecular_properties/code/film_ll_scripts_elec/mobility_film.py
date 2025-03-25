#!/usr/bin/env python
'''
从晶体结构/薄膜结构开始，完成整个计算, 后续需要把gen_sol_box.py 里面的代码merge进来
'''
import sys
import numpy as np 
import os 
from ase.data import atomic_masses, atomic_numbers, chemical_symbols  
from rdkit import Chem 
import json 
import random 
import time 
from numba import njit 
import json 
import pandas as pd 

pi_constant = 3.14159265359; hbar = 6.582119569e-16
boltzmann_constant = 8.617333262145e-5; temperature = 300 # temperature 设置为室温 

def convert_1d_to_2d(V_Ls, displaces, pair):
    V_Ls_2d = []; displaces_2d = []; mol_idx = []
    pair = pair[:,0:2] 
    for i in range(np.max(pair)+1):
        V_Ls_2d.append([]); displaces_2d.append([]); mol_idx.append([])
    # 获取2d数组 
    for idx, pp in enumerate(pair):
        V_Ls_2d[pp[0]].append(V_Ls[idx])
        displaces_2d[pp[0]].append(displaces[idx])
        mol_idx[pp[0]].append(pp[1])
    return V_Ls_2d, displaces_2d, mol_idx 


@njit 
def run_kmc(t_tot, rate_tot_list, displaces, probablity_list, mol_idx, ntot_trj):
    '''
    t_tot: 运行的总时间 单位s 
    rate_tot_list: [n_mol, 1] 每个分子可能跃迁时间的速率常数之和
    displaces: [n_mol, n_nei, 3] 每个分子跃迁的位置矢量
    probablity_list: [n_mol, n_nei] 每个分子跃迁不同位置对应的随机数分布
    mol_idx: [n_mol, n_nei] 每个分子对应的可跃迁的分子
    '''
    t_currs = []; disp_currs = []
    for i in range(ntot_trj):
        print(i)
        # we need random choose the initial mol idx each time
        idx = random.randint(0, len(displaces)-1)
        t_curr = 0; disp_curr = np.zeros(3)
        while t_curr < t_tot:
            r_num = random.uniform(0,1)
            p_tmp = probablity_list[idx]; disp_tmp = displaces[idx]; mol_idx_tmp = mol_idx[idx]
            rate_tot_tmp = rate_tot_list[idx]
            idx = np.searchsorted(p_tmp, r_num, side='left')
            disp_curr += disp_tmp[idx]
            r_num = random.uniform(1e-12,1)
            t_curr = t_curr -np.log(r_num) / rate_tot_tmp
            idx = mol_idx_tmp[idx]
        t_currs.append(t_curr); disp_currs.append(disp_curr)
    return t_currs, disp_currs

       
ntot_trj = 200; t_tot = 1e-6; dir_name = sys.argv[1]; method = sys.argv[2]
data_path = f'../../data/film_elec_mobility/{dir_name}/data_ll.npz'
data = np.load(data_path, allow_pickle=True)
reorg_e_e = data['reorg_e'] 
if method == 'OCNet':
    # mHartree
    with open(f'./infer/final_res.csv','r') as fp:
        data_ll = pd.read_csv(fp)
    val = data_ll['Val_Index'].tolist()
    V_Ls = data_ll['film_ll_pred'].tolist()
    V_Ls = np.array(V_Ls)[np.argsort(np.array(val))]
    V_Ls = np.abs(np.array(V_Ls)) / 1000. 
elif method == 'QM':
    # mHartree 
    V_Ls = np.abs(np.array(data['VL']))/1000.
elif method == 'xTB':
    # mHartree 
    V_Ls = np.abs(np.array(data['VL_xtb']))/1000. 

displaces = data['displace']; pair = data['pair']
# 将displaces & mol_idx 转换为列表格式，按[n_mol, nei_mol]格式 
V_Ls, displaces, mol_idx = convert_1d_to_2d(V_Ls, displaces, pair)

# 计算速率常数相关
rate_list_elec = []; rate_tot_list_elec = []; probablity_list_elec = [] 

def convert_data(rate_tot_list_elec,  displaces, probablity_list_elec,  mol_idx):
    # convert 
    _rate_tot_list_elec, _displaces, _probablity_list_elec, _mol_idx = [], [], [], []
    num = [len(uu) for uu in mol_idx]; num = np.max(num)
    # 避免0 
    for ii in range(len(rate_tot_list_elec)):    
        _rate_tot_list_elec.append(max(1., rate_tot_list_elec[ii]))
    for ii in range(len(mol_idx)):
        _disp = []; _prob_elec = []; _m_idx = []
        for jj in range(num):
            if jj < len(displaces[ii]):
                _disp.append(displaces[ii][jj])
                _prob_elec.append(probablity_list_elec[ii][jj])
                _m_idx.append(mol_idx[ii][jj])
            else:
                _disp.append([0.,0.,0.])
                _prob_elec.append(1.1)
                _m_idx.append(-1)
        _displaces.append(_disp)
        _probablity_list_elec.append(_prob_elec)
        _mol_idx.append(_m_idx)
    _rate_tot_list_elec = np.array(_rate_tot_list_elec, dtype=np.float32)
    
    _displaces = np.array(_displaces, dtype=np.float32)
    _probablity_list_elec = np.array(_probablity_list_elec, dtype=np.float32)
    
    _mol_idx = np.array(_mol_idx, dtype=np.int32)
    return _rate_tot_list_elec,  _displaces, _probablity_list_elec, _mol_idx


for i in range(len(V_Ls)):
    rate_list_elec.append([])
    for j in range(len(V_Ls[i])):
        tmp = V_Ls[i][j]**2 / hbar * (pi_constant / (reorg_e_e * boltzmann_constant * temperature)) ** 0.5 * \
                                    np.exp(-reorg_e_e / (4 * boltzmann_constant * temperature))
        rate_list_elec[i].append(tmp)
        #t_list[i].append(1 / rate_list[i][j])
    rate_list_elec[i] = np.array(rate_list_elec[i])
    # the time is defined as -ln(random)/rate_tot_list
    rate_tot_list_elec.append(np.sum(rate_list_elec[i]))
    p_tmp = np.cumsum(rate_list_elec[i] / np.sum(rate_list_elec[i]))
    probablity_list_elec.append(p_tmp)

rate_tot_list_elec, displaces, probablity_list_elec, mol_idx = convert_data(rate_tot_list_elec, displaces, probablity_list_elec, mol_idx) 

t_currs_elec, disp_currs_elec = run_kmc(t_tot, rate_tot_list_elec, displaces, probablity_list_elec, mol_idx, ntot_trj) 

mobility_elec = np.square(np.array(disp_currs_elec)*10**(-8.)) # cm**2 
mobility_elec = np.mean(np.sum(mobility_elec,axis=1))/t_tot/2/3/(8.617333262145e-5 * temperature)

with open(f'./{dir_name}_{method}_mob.json','w') as fp:
    json.dump({'elec':mobility_elec},fp)





