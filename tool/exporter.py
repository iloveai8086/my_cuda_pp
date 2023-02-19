# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from numpy import *
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu  # 可以这么导入的原因是因为直接就在pcdet.models里面的init里面
from pcdet.utils import common_utils

import onnx
from onnxsim import simplify
import os, sys
from exporter_paramters import export_paramters as export_paramters
from simplifier_onnx import simplify_onnx as simplify_onnx


#     基本预处理：生成柱体。
#     预处理：生成 BEV 特征图（10 个通道）。  这个十个通道就是xyzi + 底面中心偏移和均值偏移的十维特征
#     用于 TensorRT 的 ONNX 模型：通过 TensorRT 实现的 ONNX 模式。
#     后处理：通过解析 TensorRT 引擎输出生成边界框。

#   基本预处理步骤将点云转换为基本特征图。基本特征图包含以下组成部分：
#   基本特征图。柱体坐标：每根柱体的坐标。参数：柱体数量。

# 用于TensorRT的ONNX模型
# 出于以下原因修改 OpenPCDet 的原生点柱：
# 小型操作过多，并且内存带宽低。
# NonZero 等一些 TensorRT 不支持的操作。
# ScatterND 等一些性能较低的操作。
# 使用“dict”作为输入和输出，因此无法导出 ONNX 文件。
# 为了从原生 OpenPCDet 导出 ONNX，我们修改了该模型。

# 您可把整个 ONNX 文件分为以下几个部分：
# 输入：BEV 特征图、柱体坐标、参数，均在预处理中生成。
# 输出：类、框、Dir_class，在后处理步骤中解析后生成一个边界框。
# ScatterBEV：将点柱（一维）转换为二维图像，可作为 TensorRT 的插件。
# 其他：TensorRT 支持的其他部分。

# 看了下onnx的netron:他可能linear全连接层用的是matmul，然后输入的名字和输出的名字就是下面那个api调用的
# 可以去这个地址看看onnx的一些算子长什么样  https://blog.csdn.net/mzpmzk/article/details/120293992
# 关于onnxsim这些东西可以去看这个博客   https://zhuanlan.zhihu.com/p/350702340

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar_port_5cls.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../datas',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default="checkpoint_epoch_72_5cls.pth", help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    export_paramters(cfg)
    logger = common_utils.create_logger()
    logger.info('------ Convert OpenPCDet model for TensorRT ------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                          dataset=demo_dataset)  # 这边都是参考了pcdet的demo
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    np.set_printoptions(threshold=np.inf)  # 意思是输出数组的时候完全输出，不需要省略号将中间数据省略
    with torch.no_grad():
        MAX_VOXELS = 10000

        batch_size = 1

        dummy_voxel_features = torch.zeros(
            (MAX_VOXELS, 32, 4),
            dtype=torch.float32,
            device='cuda:0')  # （10000，32，4）  BEV 特征图

        dummy_voxel_num_points = torch.zeros(
            (MAX_VOXELS),
            dtype=torch.float32,
            device='cuda:0')  # （10000，）  参数？ 最大柱子的个数 实际看onnx的时候好像也不是这个三个shape

        dummy_coords = torch.zeros(
            (MAX_VOXELS, 4),
            dtype=torch.float32,
            device='cuda:0')  # （10000，4）  柱体的坐标
        # 直接加载模型，导出为onnx
        # from torch.onnx.symbolic_helper import
        # 这里注意的是，onnx export函数需要加上keep_initializers_as_inputs=True, opset_version=11两个参数，
        # 否则后面onnx-simplifier可能会段错误，无法简化模型
        torch.onnx.export(model,  # model being run  我要导出的model
                          (dummy_voxel_features, dummy_voxel_num_points, dummy_coords),  # 我目前的理解就是把模型的输入拆分成这几个
                          # 模型的输入, 任何非Tensor参数都将硬编码到导出的模型中；任何Tensor参数都将成为导出的模型的输入，
                          # 并按照他们在args中出现的顺序输入。
                          # model input (or a tuple for multiple inputs)
                          "./pointpillar.onnx",  # where to save the model (can be a file or file-like object) 保存
                          export_params=True,  # store the trained parameter weights inside the model file
                          # 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          # 是否使用“常量折叠”优化。常量折叠将使用一些算好的常量来优化一些输入全为常量的节点。
                          keep_initializers_as_inputs=True,  # 可能就是为了simplify做的
                          input_names=['input', 'voxel_num_points', 'coords'],  # the model's input names
                          output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],  # the model's output names
                          )
        # 我自己确实做了简化，我发现没简化之前的操作就是很多常量的计算，开启了do_constant_folding，很多常量就被合并了
        onnx_model = onnx.load("./pointpillar.onnx")  # load onnx model
        # 一般来说，导出到onnx之后，会有一些冗余的操作，需要simplify简化操作，比如identity层，bn和relu融合
        # # 基于onnx-simplifier简化模型，         https://github.com/daquexian/onnx-simplifier
        # # 也可以命令行输入python3 -m onnxsim input_onnx_model output_onnx_model
        # # 或者使用在线网站直接转换https://convertmodel.com/
        # 第三种在线调用方便，还能将onnx模型转换为ncnn, mnn等模型格式。

        # onnx-simplifier对于高版本pytorch不那么支持，转换可能失败，所以设置skip_fuse_bn=True跳过融合bn层。
        # 这种情况下onnx-simplifier转换出来的onnx模型可能比转换前的模型大，原因是补充了shape信息。
        # simplify(onnx_model, skip_fuse_bn=True)
        model_simp, check = simplify(onnx_model)  # onnxsim 原生的接口   一般就是直接导出完的模型再去load，
        # 然后得到就是简化后的模型和check，融合BN relu等
        assert check, "Simplified ONNX model could not be validated"

        model_simp = simplify_onnx(model_simp)
        onnx.save(model_simp, "pointpillar.onnx")
        print("export pointpillar.onnx.")
        print('finished exporting onnx')

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
