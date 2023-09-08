# %% [markdown]
# # 观修罗

# %% [markdown]
# ## 导入相应库

# %%
import os
import shutil
import json5
import argparse
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn


# %%
# -------------------工 具 函 数-------------------
from utils.m_fmtjson import m_fmtjson
from utils.m_logger import m_logger
from utils.get_Himawari_file import get_Himawari_file
from utils.m_processV2 import demon
from utils.get_Cld2NcP import get_Cld2NcP


# %% [markdown]
# ## 加载参数

# %%
parser = argparse.ArgumentParser(description='config parameters')
parser.add_argument('--config', type=str, default='./CldNetV2_inference_online.jsonc', help="Program running parameters")
parser.add_argument('--key', type=str, default='001001', help="Program running parameters")
args, unknown = parser.parse_known_args()
f_config_file = args.config
f_config_key = args.key
with open(f_config_file, "r+", encoding="utf-8") as fp:
    f_config = json5.load(fp)[f_config_key]
# 创建输出文件夹
f_config["out_dir"] = f_config["out_dir"]+f_config_key+"/"

if os.path.exists(f_config["out_dir"]):
    if f_config["is_remove"]:                           # 删除输出文件夹
        shutil.rmtree(f_config["out_dir"])
        print('clear outdir!')
        os.makedirs(f_config["out_dir"])
else:
    os.makedirs(f_config["out_dir"])

if "checkpoint" not in f_config or f_config["checkpoint"] is None:
    f_config["checkpoint"] = f_config["out_dir"]+f"{f_config_key}_{f_config['kf_idx']}_checkpoint_best.pth.tar"

f_config_fmt = m_fmtjson(f_config)
if f_config["show_log"]:
    print(f_config_fmt)

# -------------------定义logger-------------------
log_file = os.path.join(f_config["out_dir"], f_config_key+'.log')
logger = m_logger(log_file, f_config["logging"])
# 储存模型参数
logger.info("-------------------Model config parameters-------------------")
logger.info(f_config_fmt)
logger.info("-------------------Model config parameters-------------------")

logger.info("Version of PyTorch: {0}".format(torch.__version__))
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = f_config["device"]
logger.info(f"Using {device} device")


# %% [markdown]
# ## 导入数据加载函数

# %%
from utils.load_dataV2 import load_data, Dataset


# %% [markdown]
# ## 云属性推理函数

# %%
def get_Cld(load_data_params, model_daytime, model_nighttime, sign):
    dataset_daytime = Dataset(
        load_data_params,
        mask_bands=f_config["load_data"]["mask_bands"] if 'mask_bands' in f_config["load_data"] else None,
        mask_ratio=-1.0,
        stage='inference'
    )
    dataLoader_daytime = torch.utils.data.DataLoader(
        dataset_daytime,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    dataset_nighttime = Dataset(
        load_data_params,
        mask_bands=f_config["load_data"]["mask_bands"] if 'mask_bands' in f_config["load_data"] else None,
        mask_ratio=1.0,
        stage='inference'
    )
    dataLoader_nighttime = torch.utils.data.DataLoader(
        dataset_nighttime,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    # -------------------推理获得结果-------------------
    _, output_daytime, _, _ = demon(
        model_daytime,
        dataLoader_daytime,
        logger=logger,
        log_level=10,
        target_names=["x0"] if "target_name" not in f_config else f_config["target_name"],
        hw=[5, 5] if "hw" not in f_config else f_config["hw"],
        nbs=1,
        pre_save=True,
        is_train=False
    )

    _, output_nighttime, _, _ = demon(
        model_nighttime,
        dataLoader_nighttime,
        logger=logger,
        log_level=10,
        target_names=["x0"] if "target_name" not in f_config else f_config["target_name"],
        hw=[5, 5] if "hw" not in f_config else f_config["hw"],
        nbs=1,
        pre_save=True,
        is_train=False
    )

    # 保存变量
    Cld = []
    Cld_encoding = dict()
    clearsky_sign = None
    for xname in f_config["target_name"]:
        # print('-'*60, xname, '-'*60)
        # logger.info(f"{'-'*60+xname+'-'*60}")
        if xname == "CLTYPE":
            temp = torch.argmax(torch.softmax(output_daytime[xname][0], dim=1), dim=1)[0].cpu().numpy().astype("float32")
            clearsky_sign = temp == 0
        else:
            vr = pd.read_csv(f_config["load_data"]["params"]["vr_file"], index_col=0)
            xmin = vr[xname]["min"]
            xmax = vr[xname]["max"]
            temp = output_daytime[xname][0][0, 0].cpu().numpy().astype("float32")*(xmax-xmin)+xmin
            if clearsky_sign is not None:
                temp[clearsky_sign] = np.nan
            temp[temp<=0] = np.nan
        temp_daytime = temp

        if xname == "CLTYPE":
            temp = torch.argmax(torch.softmax(output_nighttime[xname][0], dim=1), dim=1)[0].cpu().numpy().astype("float32")
            clearsky_sign = temp == 0
        else:
            vr = pd.read_csv(f_config["load_data"]["params"]["vr_file"], index_col=0)
            xmin = vr[xname]["min"]
            xmax = vr[xname]["max"]
            temp = output_nighttime[xname][0][0, 0].cpu().numpy().astype("float32")*(xmax-xmin)+xmin
            if clearsky_sign is not None:
                temp[clearsky_sign] = np.nan
            temp[temp<=0] = np.nan
        temp_nighttime = temp

        temp_daytime[sign] = temp_nighttime[sign]
        temp = temp_daytime
        lon = np.linspace(80, 200, temp.shape[1], endpoint=False)
        lat = np.linspace(60, -60, temp.shape[0], endpoint=False)
        
        Cld.append(
            xr.DataArray(
                data=temp,
                dims=("latitude", "longitude"),
                coords={"latitude": lat, "longitude": lon},
                name=xname,
                attrs=get_Cld2NcP(xname, flag='attrs')
            )
        )
        Cld_encoding[xname] = get_Cld2NcP(xname, flag='encoding')

    Cld = xr.merge(Cld)

    return Cld, Cld_encoding


# %% [markdown]
# ## 推理过程

# %%
if "target_name" in f_config and (not isinstance(f_config["target_name"], (list, tuple))):
    f_config["target_name"] = (f_config["target_name"],)

if 'go_inference' not in f_config or f_config['go_inference']:
    # 获取RS阈值
    with open(f_config["load_data"]["params"]["RS_Threshold_file"], "r+", encoding="utf-8") as fp:
        RS_Threshold = json5.load(fp)
    f_config["load_data"]["params"]["RS_Threshold"] = RS_Threshold

    # -----------------------------------daytime
    # 加载模型参数
    checkpoint_daytime = torch.load(f_config["checkpoint"]["daytime"])
    # 实例化模型
    model_daytime = checkpoint_daytime["model"]
    model_daytime = model_daytime.to(device)

    # -----------------------------------nighttime
    # 加载模型参数
    checkpoint_nighttime = torch.load(f_config["checkpoint"]["nighttime"])
    # 实例化模型
    model_nighttime = checkpoint_nighttime["model"]
    model_nighttime = model_nighttime.to(device)

    # 生成RS文件列表
    ss_time = pd.date_range(
        start=f_config["load_data"]["params"]["start_time"],
        end=f_config["load_data"]["params"]["end_time"],
        freq=f_config["load_data"]["params"]["freq"]
    )
    RS_dir = f_config["load_data"]["params"]["RS_dir"]
    Cld_dir = f_config["load_data"]["params"]["Cld_dir"]
    for tt in ss_time if ("flashback" not in f_config) or (not f_config["flashback"]) else ss_time[::-1]:
        # logger.info(f"{'-'*60}{tt}{'-'*60}")
        # -------------------获取遥感文件-------------------
        Himawari_flag = "H08" if 'Himawari_flag' not in f_config["load_data"]["params"] else f_config["load_data"]["params"]['Himawari_flag']
        H08_file = get_Himawari_file(
            tt,
            Himawari_flag=Himawari_flag,
            resolution=0.05 if 'resolution' not in f_config["load_data"]["params"] else f_config["load_data"]["params"]['resolution'],
            sign='RS'
        )
        RS_file = RS_dir+H08_file
        if not os.path.exists(RS_file):
            Himawari_flag = "H09"
            H08_file = get_Himawari_file(
                tt,
                Himawari_flag=Himawari_flag,
                resolution=0.05 if 'resolution' not in f_config["load_data"]["params"] else f_config["load_data"]["params"]['resolution'],
                sign='RS'
            )
            RS_file = RS_dir+H08_file
            if not os.path.exists(RS_file):
                continue

        # -------------------构建输出目标文件名-------------------
        H08_file = get_Himawari_file(
            tt,
            Himawari_flag=Himawari_flag,
            resolution=0.05 if 'resolution' not in f_config["load_data"]["params"] else f_config["load_data"]["params"]['resolution'],
            sign='cloud'
        )
        Cld_file = f_config["out_dir"] + H08_file

        sign = xr.open_dataset(RS_file).variables["SOZ"][:-1, :-1].values > f_config["SOZ"]
        # -------------------构建数据加载器-------------------
        load_data_params = f_config["load_data"]["params"].copy()
        load_data_params["RS_files"] = [RS_file]
        # ---------------------云属性推理---------------------
        Cld, Cld_encoding = get_Cld(load_data_params, model_daytime, model_nighttime, sign)
        
        Cld.attrs = get_Cld2NcP('Cloud', flag='attrs')
        Cld.to_netcdf(Cld_file, encoding=Cld_encoding)
        # logger.info(f"Saved successfully......")
        logger.info(f"{'-'*30}{tt}{'-'*30}")
