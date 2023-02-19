/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef _BEV_H_
#define _BEV_H_

#include <vector>
#include <string>

#include <cuda_runtime_api.h>

#include "NvInferPlugin.h"

namespace nvinfer1
{
namespace plugin
{

class ScatterBevPlugin : public nvinfer1::IPluginV2DynamicExt // 这边就是实际的继承了IPluginV2DynamicExt
{
public:

    ScatterBevPlugin(); // 这边的两个构造函数一个是为了创建和clone的时候使用的，一个是在反序列化的时候使用的

    ScatterBevPlugin(const void* data, size_t length);

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;  // clone
	// 这个就是输出的维度   还有个输出的type
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, 
        const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, 
        int nbInputs, int nbOutputs) noexcept override;  // 通过判断POS索引的输入输出是否支持inOut[pos].format inout[pos].type指定的格式和类型

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;  //检查函数，判断输入和输出的类型数量是否正确

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;  //运行时的相关函数，tmp什么的，看我的笔记，给个4-5G？不浪费时间和显存复用
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, 
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, 
        void* workspace, cudaStream_t stream) noexcept override;  // 实际推理的函数，输入描述、输出描述。

    // IPluginV2Ext Methods    就在这里了，输出的type的定义，包括什么float half啥的 这些类型都是需要自己指定的。
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, 
        int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;  // type和version必须是唯一的

    const char* getPluginVersion() const noexcept override;

    int getNbOutputs() const noexcept override; // 获得layer的输出个数

    int initialize() noexcept override; // 初始化函数，一般在run之前就开始执行了。申请权值显存空间并copy值。

    void terminate() noexcept override;  // 释放上面开辟的空间

    size_t getSerializationSize() const noexcept override; // 返回序列化时需要写多少字节到buffer中

    void serialize(void* buffer) const noexcept override;  // 序列化函数，将plugin的参数权值写入到buffer里面

    void destroy() noexcept override;  //不常用，

    void setPluginNamespace(const char* pluginNamespace) noexcept override; //

    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;

    // the num -- output channels size of the 2D backbone network
    const int featureNum_ = 64;

    // the y -- output size of the 2D backbone network
    const int feature_y_size_ = 496;

    // the x -- output size of the 2D backbone network
    const int feature_x_size_ = 864;

    void *cacheBEV_ = nullptr;
};

class ScatterBevPluginCreator : public nvinfer1::IPluginCreator
{
public:
    ScatterBevPluginCreator();

    const char* getPluginName() const noexcept override; // 获得name和version，就是为了辨识creator

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;  //PluginFieldCollection 创建plugin，

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;  // 将op需要的权重和参数一个一个提取出来，然后调用上文提到的第一个构造函数

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;  // 反序列化，调用反序列化的构造函数，生成plugin

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;

    static std::vector<nvinfer1::PluginField> mPluginAttributes;

    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
