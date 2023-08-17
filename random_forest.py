
"""
This contains implementations of NAS-Bench-201 architecture conversions based on
https://github.com/automl/NASLib/blob/zerocost/naslib/search_spaces/nasbench201/conversions.py
"""


import argparse
from inspect import isasyncgenfunction
import torch
import json
import os, sys, glob
import pickle
import json
import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import plotly.express as px
import plotly
import re
import csv
from plotly.subplots import make_subplots
import plotly.graph_objs as go


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn import metrics


from robustness_dataset import RobustnessDataset


##############################################################################
#
#                              Arguments
#
##############################################################################

parser = argparse.ArgumentParser(description='Args for ZCP evals of Robustness Dataset')
parser.add_argument('--image_data',                 type=str, default='ImageNet16-120', help="Choices between cifar10, cifar100 and ImageNet16-120")
parser.add_argument("--robustness_data_path",       type=str, default="robustness-data")
parser.add_argument("--zcp_data_path",              type=str, default="zcp_data/zc_nasbench201.json")
parser.add_argument("--regression_target_1",        type=str, default="clean")
parser.add_argument("--regression_target_2",        type=str, default="fgsm@Linf, eps=1.0", help="fgsm, pgd, aa_apgd-ce, aa_square")

args = parser.parse_args()


##############################################################################
colors = ["EE2967", "A93C93", "0071BC", "3FBC9D", "F7965A"]

map_color = {  
    "plain" : "#A93C93", 
    "grasp": "#0071BC", 
    "l2_norm": "#A93C93", 
    "hessian": "#3FBC9D", 
    "fisher": "#0071BC", 
    "epe_nas": "#EE2967", 
    "grad_norm": "#EE2967",
    "snip":  "#0071BC",
    "zen": "#F7965A",
    "params": "#A93C93",
    "flops": "#A93C93",
    "synflow": "#0071BC",
    "jacob_fro": "#EE2967", 
    "nwot": "#EE2967", 
    "jacov": "#EE2967"
    }
    
map_symbol = {  
    "plain" : "\\blacksquare", 
    "grasp": "\\blacklozenge", 
    "l2_norm": "\\blacksquare", 
    "hessian": "\\bullet", 
    "fisher": "\\blacklozenge", 
    "epe_nas": "\\bigstar", 
    "grad_norm": "\\bigstar",
    "snip":  "\\blacklozenge",
    "zen": "\\blacktriangledown",
    "params": "\\blacksquare",
    "flops": "\\blacksquare",
    "synflow": "\\blacklozenge",
    "jacob_fro": "\\bigstar", 
    "nwot": "\\bigstar", 
    "jacov": "\\bigstar"
    }    

##############################################################################




OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
OP_NAMES_NB201 = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']

EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
OPS_TO_NB201 = {
    "AvgPool1x1": "avg_pool_3x3",
    "ReLUConvBN1x1": "nor_conv_1x1",
    "ReLUConvBN3x3": "nor_conv_3x3",
    "Identity": "skip_connect",
    "Zero": "none",
}

def convert_op_indices_to_str(op_indices):
    edge_op_dict = {
        edge: OP_NAMES_NB201[op] for edge, op in zip(EDGE_LIST, op_indices)
    }

    op_edge_list = [
        "{}~{}".format(edge_op_dict[(i, j)], i - 1)
        for i, j in sorted(edge_op_dict, key=lambda x: x[1])
    ]

    return "|{}|+|{}|{}|+|{}|{}|{}|".format(*op_edge_list)
    
# Load NAS-Bench-Suite ZCP data
with open(args.zcp_data_path, 'r') as f:
    zcp_data = json.load(f)

# Load Robustness Dataset data
rob_data = RobustnessDataset(path=args.robustness_data_path)
results = rob_data.query(
    # data specifies the evaluated dataset
    data = args.image_data,
    # measure specifies the evaluation type
    measure = "accuracy", 
    # key specifies the attack types
    key = RobustnessDataset.keys_clean + RobustnessDataset.keys_adv + RobustnessDataset.keys_cc
)


# Pre-calculated Jacobian Robustness Dataset, Jung et at., ICLR 2023
print("Get Jacobian ZCP")
jac_paths = glob.glob("robustness_dataset_zcp/Jacobian_results_restructured/"+args.image_data+"/*/jacobian_train_random.torch")

jacobian_results = {}

for file in tqdm.tqdm(jac_paths):
    idx =  file.split(os.path.sep)[-2]
    jacob_proxy = torch.load(file)
    jacobian_results[idx] = list(jacob_proxy.values())[0][0] # first of 10 batches


# Pre-calculated Hessian Robustness Dataset, Jung et at., ICLR 2023
if args.image_data=="cifar10":
    print("Get Hessian ZCP")
    hess_paths = glob.glob("robustness_dataset_zcp/Hessian_results_restructured/"+args.image_data+"/*/hessian_train_random.torch")

    hessian_results = {}

    for file in tqdm.tqdm(hess_paths):
        idx =  file.split(os.path.sep)[-2]
        hessian_proxy = torch.load(file)
        hessian_results[idx] = list(hessian_proxy.values())[0]

##############################################################################
#
#                              Create Data
#
##############################################################################

print("Combine all Data")
all_data = {d:{} for d in ['clean', 'aa_apgd-ce@Linf', 'aa_square@Linf', 'fgsm@Linf', 'pgd@Linf'] |zcp_data[args.image_data]['(3, 4, 4, 4, 2, 4)'].keys()}


del all_data["id"]

all_data["jacob_fro"]={}

if args.image_data=="cifar10":
    all_data["hessian"]={}
        
    for op_indices in tqdm.tqdm(zcp_data[args.image_data]):
        string = convert_op_indices_to_str(eval(op_indices))
        nb201_id = rob_data.string_to_id(string)
        if nb201_id in rob_data.non_isomorph_ids:
            idx = rob_data.get_uid(nb201_id)
            if idx not in hessian_results.keys():
                    continue
            for zcp in ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen' ]:
                all_data[zcp][idx] = zcp_data[args.image_data][op_indices][zcp]["score"]
            all_data['val_accuracy'][idx] = zcp_data[args.image_data][op_indices]["val_accuracy"]
            for attack in ['clean', 'aa_apgd-ce@Linf', 'aa_square@Linf', 'fgsm@Linf', 'pgd@Linf']:
                all_data[attack][idx] = results[args.image_data][attack]["accuracy"][idx]
            all_data["hessian"][idx] = hessian_results[idx] 
            all_data["jacob_fro"][idx] = jacobian_results[idx] 
        else:
            continue

else:
    for op_indices in tqdm.tqdm(zcp_data[args.image_data]):
        string = convert_op_indices_to_str(eval(op_indices))
        nb201_id = rob_data.string_to_id(string)
        if nb201_id in rob_data.non_isomorph_ids:
            idx = rob_data.get_uid(nb201_id)
            for zcp in ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen' ]:
                all_data[zcp][idx] = zcp_data[args.image_data][op_indices][zcp]["score"]
            all_data['val_accuracy'][idx] = zcp_data[args.image_data][op_indices]["val_accuracy"]
            for attack in ['clean', 'aa_apgd-ce@Linf', 'aa_square@Linf', 'fgsm@Linf', 'pgd@Linf']:
                all_data[attack][idx] = results[args.image_data][attack]["accuracy"][idx]
            all_data["jacob_fro"][idx] = jacobian_results[idx] 
        else:
            continue


##############################################################################
#
#                              Create Dataframe
#
##############################################################################
df = pd.DataFrame()
df["id"] = list(all_data["grad_norm"].keys())
if args.image_data=="cifar10":
    for column in ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen', "hessian", "jacob_fro",\
                   "clean", "val_accuracy"]:
        if not (df["id"] == list(all_data[column].keys())).all():
            print(column)
            break
        df[column] = list(all_data[column].values())
else:
    for column in ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip', 'synflow', 'zen', "jacob_fro",\
                   "clean", "val_accuracy"]:
        if not (df["id"] == list(all_data[column].keys())).all():
            print(column)
            break
        df[column] = list(all_data[column].values())

for i,e  in enumerate(rob_data.meta["epsilons"]["fgsm@Linf"]):
    if not (df["id"] == list(all_data["fgsm@Linf"].keys())).all():
        break
    df["fgsm@Linf, eps="+str(e)] = np.array(list(all_data["fgsm@Linf"].values()))[:,i]

for i,e  in enumerate(rob_data.meta["epsilons"]["pgd@Linf"]):
    if not (df["id"] == list(all_data["pgd@Linf"].keys())).all():
        break
    df["pgd@Linf, eps="+str(e)] = np.array(list(all_data["pgd@Linf"].values()))[:,i]

for i,e  in enumerate(rob_data.meta["epsilons"]["aa_apgd-ce@Linf"]):
    if not (df["id"] == list(all_data["aa_apgd-ce@Linf"].keys())).all():
        break
    df["aa_apgd-ce@Linf, eps="+str(e)] = np.array(list(all_data["aa_apgd-ce@Linf"].values()))[:,i]

for i,e  in enumerate(rob_data.meta["epsilons"]["aa_square@Linf"]):
    if not (df["id"] == list(all_data["aa_square@Linf"].keys())).all():
        break
    df["aa_square@Linf, eps="+str(e)] = np.array(list(all_data["aa_square@Linf"].values()))[:,i]



if args.image_data=="cifar10":
    
    x_labels  = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip',\
         'synflow', 'zen', "hessian", "jacob_fro"]
else:
    x_labels  = ['epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain', 'snip',\
         'synflow', 'zen', "jacob_fro"]

y_labels = ['clean','fgsm@Linf, eps=0.1', 'fgsm@Linf, eps=0.5', 'fgsm@Linf, eps=1.0','fgsm@Linf, eps=2.0', 'fgsm@Linf, eps=3.0', 'fgsm@Linf, eps=4.0','fgsm@Linf, eps=5.0', 'fgsm@Linf, eps=6.0', 'fgsm@Linf, eps=7.0',
       'fgsm@Linf, eps=8.0', 'fgsm@Linf, eps=255.0', 'pgd@Linf, eps=0.1', 'pgd@Linf, eps=0.5', 'pgd@Linf, eps=1.0', 'pgd@Linf, eps=2.0',
       'pgd@Linf, eps=3.0', 'pgd@Linf, eps=4.0', 'pgd@Linf, eps=8.0',
       'aa_apgd-ce@Linf, eps=0.1', 'aa_apgd-ce@Linf, eps=0.5',
       'aa_apgd-ce@Linf, eps=1.0', 'aa_apgd-ce@Linf, eps=2.0', 'aa_apgd-ce@Linf, eps=3.0', 'aa_apgd-ce@Linf, eps=4.0',
       'aa_apgd-ce@Linf, eps=8.0', 'aa_square@Linf, eps=0.1', 'aa_square@Linf, eps=0.5', 'aa_square@Linf, eps=1.0',
       'aa_square@Linf, eps=2.0', 'aa_square@Linf, eps=3.0','aa_square@Linf, eps=4.0', 'aa_square@Linf, eps=8.0']


##############################################################################
#
#             Feature importance based on mean decrease in impurity
#
##############################################################################

def add_symbol(text: str, color: str, symbol: str):
    return f"$\\text{{{text}}}\\color{{{color}}}{{{symbol}}}$"


def get_feature_importance(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators = 100,
                            n_jobs = -1,
                            oob_score = True,
                            bootstrap = True,
                            random_state = 42)


    rf.fit(X_train, y_train)
    print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(rf.score(X_train, y_train), 
                                                                                                  rf.oob_score_,
                                                                                                 rf.score(X_test, y_test)))


    feature_names = [X.columns[i] for i in range(X.shape[1])]

    ##############################################################################
    #
    #             Feature importance based on mean decrease in impurity
    #
    ##############################################################################

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=True)


    ##############################################################################
    #
    #    Feature importance based on mean decrease in impurity Plot
    #
    ##############################################################################
    forest_importances = forest_importances.to_frame().reset_index().rename({0: "score", "index":"proxy"}, axis=1)
    forest_importances = forest_importances[["proxy", "score"]]
    forest_importances.groupby("proxy")
    idx = [v[0] for k, v in forest_importances.groupby(["proxy"], group_keys=False).groups.items()]
    forest_importances = forest_importances.iloc[idx]

    forest_importances= forest_importances.sort_values(by="score", ascending=True)

    forest_importances['proxy'] = forest_importances.apply(lambda row: add_symbol(row['proxy'], map_color[row['proxy']], map_symbol[row['proxy']]),axis=1)
    return forest_importances

##############################################################################
#
#                              Random Forest  
#
##############################################################################

header = ['target', 'train mse', 'train R2', "test mse", "test R2", "OOB"]

# Save results 

print("Regression for: {} and {}".format(args.regression_target_1, args.regression_target_2))    
X = df[x_labels]
y = df[[args.regression_target_1, args.regression_target_2]]
forest_importances_1 = get_feature_importance(X,y)


print("Regression for: {}".format(args.regression_target_1))    
X = df[x_labels]
y = df[args.regression_target_1]
forest_importances_2 = get_feature_importance(X,y)
forest_importances_2.rename(columns = {"score": "score_target_1"}, inplace=True)


print("Regression for: {}".format(args.regression_target_2))    
X = df[x_labels]
y = df[args.regression_target_2]
forest_importances_3 = get_feature_importance(X,y)
forest_importances_3.rename(columns = {"score": "score_target_2"}, inplace=True)

data_all = forest_importances_1.append(forest_importances_2, ignore_index=True).append(forest_importances_3, ignore_index=True)

fig_1 = go.Bar(x=data_all["score"], y=data_all["proxy"], orientation="h", name=str(args.regression_target_1)+" and "+ str(args.regression_target_2) )
fig_2 = go.Bar(x=data_all["score_target_1"], y=data_all["proxy"], orientation="h", name=str(args.regression_target_1))
fig_3 = go.Bar(x=data_all["score_target_2"], y=data_all["proxy"], orientation="h", name=str(args.regression_target_2))

fig = make_subplots()
fig.add_trace(fig_1)
fig.add_trace(fig_2)
fig.add_trace(fig_3)

fig.update_layout(legend=dict(
    yanchor="bottom",
    y=0.01,
    xanchor="right",
    x=0.99
))

fig.update_layout(
    plot_bgcolor='white',
    title={
        'text': "Feature importances using MDI - "+str(args.regression_target_1)+" and "+str(args.regression_target_2) , 
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        yaxis_title= "Mean decrease in impurity"

)

fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
)

fig.write_image("plots/"+str(args.image_data)+"_mid_importance_"+args.regression_target_1+args.regression_target_2+".pdf")




