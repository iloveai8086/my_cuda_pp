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
#include <cassert>
#include <iostream>
#include "ScatterBEV.h"
#include "ScatterBEV_kernels.h"
/**
 * For the usage of those member function, please refer to the
 * offical api doc.
 * https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_ext.html
 */

#ifndef CUTEDEBUG
#define CUTEDEBUG 0 // set debug mode, if you want to see the api call, set it to 1
#endif

#if CUTEDEBUG
#define cutelog(...) {\
    char str[100];\
    sprintf(str, __VA_ARGS__);\
    std::cout << " (๑¯◡¯๑) noexcept CUSTOM PLUGIN TRACE----> call " << "[" << __FILE__ << "][" \
              << __FUNCTION__ << "][Line " << __LINE__ << "] " << str << std::endl;\
    }
#else
#define cutelog(...)
#endif

using namespace nvinfer1;
using nvinfer1::plugin::ScatterBevPlugin;
using nvinfer1::plugin::ScatterBevPluginCreator;

static const char* PLUGIN_VERSION{"1"};  // 定义了全局的 plugin的version 和 name
static const char* PLUGIN_NAME{"ScatterBEV"};

// Static class fields initialization    这边是creator的静态变量，不需要管直接复制过来就行了，和trt5.1的bert的demo是一模一样的
PluginFieldCollection ScatterBevPluginCreator::mFC{};
std::vector<PluginField> ScatterBevPluginCreator::mPluginAttributes;

// 在bert的demo里面还有一个注册的过程如下
// REGISTER_TENSORRT_PLUGIN(ScatterBevPluginCreator)

ScatterBevPlugin::ScatterBevPlugin()  // 我看bert这边有什么是否支持fp16，然后什么权值啥的也传进来了，这边怎么什么都没有？
{
    cutelog("wow I run to here now");
}

ScatterBevPlugin::ScatterBevPlugin(const void* data, size_t length)  // 反序列化的构造函数，当时怎么把type按照顺序写到文件里面，就按顺序读就行了，好像是正序的，不排除坑是反序的。
{
    cutelog("wow I run to here now");
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* ScatterBevPlugin::clone() const noexcept
{
    cutelog("wow I run to here now");
    auto* plugin = new ScatterBevPlugin(*this);  // new 一个构造函数，就是一个赋值构造函数
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}


// 得到输出，根据我实际的计算返回，有时会做了一些对输入维度的检查，我个人觉得可以结合Python那边的代码看维度的变换
nvinfer1::DimsExprs ScatterBevPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    cutelog("wow I run to here now");
    assert(outputIndex == 0);
    for(int i=0;i<nbInputs;i++) {
        printf("input[%d]: ", i);
        for(int j=0;j<inputs[i].nbDims;j++) {
            printf("%d ", inputs[i].d[j]->getConstantValue());
        }
        printf("\n");
    }
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = exprBuilder.constant(1);
    output.d[1] = exprBuilder.constant(featureNum_);
    output.d[2] = exprBuilder.constant(feature_y_size_);
    output.d[3] = exprBuilder.constant(feature_x_size_);

    return output;
}


//  为了让plugin更加的健壮，判断输入输出的类型什么的
bool ScatterBevPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{

    cutelog("wow I run to here now");
    assert(nbInputs == 3);
    assert(nbOutputs == 1);

    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void ScatterBevPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
  cutelog("wow I run to here now");
}


// 这边相较于传统的static的模式，多了一个输入和输出的维度的描述
size_t ScatterBevPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
  cutelog("wow I run to here now");
  unsigned int cacheBEVSize = inputs[0].dims.d[0]
                                * inputs[0].dims.d[2] * sizeof(float);
  return cacheBEVSize;
}


// 真正的执行的函数，就是获得输入，然后走kernel
int ScatterBevPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
  cutelog("wow I run to here now");

  unsigned int batch = 1;
  unsigned int featureNum = featureNum_;
  unsigned int featureY = feature_y_size_;
  unsigned int featureX = feature_x_size_;
  
  std::cout<<featureNum<<' '<<featureY<<' '<<featureX<<std::endl;
  
  const float *in = (const float *)inputs[0];  // 实际拿到输入的维度
  const float *coords_data = (const float *)(inputs[1]);
  const unsigned int *params_data = (const unsigned int *)(inputs[2]);
  float *spatial_feature_data = (float *)(outputs[0]);

  unsigned int count = inputDesc[0].dims.d[0];
  cacheBEV_ = workspace;
  const float *pillar_features_data = (const float *)(cacheBEV_);

  //cudaMemcpyAsync(paramsPtr, params_data, 5*sizeof(int), cudaMemcpyDefault, stream);

  checkCudaErrors(cudaMemsetAsync(spatial_feature_data, 0, batch*featureNum*featureY*featureX * sizeof(float), stream));
  checkCudaErrors(reduceMax_kernel_launcher((const float*)in, (float*)pillar_features_data, count, stream));
  checkCudaErrors(scatterBEV_kernel_launcher(pillar_features_data, coords_data, params_data, featureX, featureY, spatial_feature_data, stream));

  return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType ScatterBevPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    cutelog("wow I run to here now");
    return inputTypes[0];  // 直接返回in的data type，bert那边就是   (mType == DataType::KHALF || mType == DataType::KFLOAT) return mtype; 这种写法
}

// IPluginV2 Methods

const char* ScatterBevPlugin::getPluginType() const noexcept
{
    cutelog("wow I run to here now");
    return PLUGIN_NAME;  // 这两就直接是返回了name和version
}

const char* ScatterBevPlugin::getPluginVersion() const noexcept
{
    cutelog("wow I run to here now");
    return PLUGIN_VERSION;
}

int ScatterBevPlugin::getNbOutputs() const noexcept  // 输出就是一个
{
    cutelog("wow I run to here now");
    return 1;
}

int ScatterBevPlugin::initialize() noexcept  //bert那边有一些申请了显存空间的，为什么， 这边什么都没写，cudamalloc cudMemcpy，在enqueue里面不推荐使用cudamalloc的，但是这边用了
// 那是因为我们在enqueue里面的时候，用了workspace size来复用这个显存，但是权值和参数是没法什么复用的，就是初始化就一直在，除非这个对象被销毁了
{
    cutelog("wow I run to here now");
    return 0;
}

void ScatterBevPlugin::terminate() noexcept  // 一定要释放，不然就是显存泄露
{
    cutelog("wow I run to here now");
}

size_t ScatterBevPlugin::getSerializationSize() const noexcept  //好像是这边没注册还是啥的，或者是onnx推理，这边根本就没有什么序列化的大小？正常应该是就是  多少个权值，参数啥的，sizeof（float） * 
{
    cutelog("wow I run to here now");
    return 0;
}

void ScatterBevPlugin::serialize(void* buffer) const noexcept  //上面的函数把实际的大小算好了，那么开始写了，然后写完看看和我实际的算的是不是一样的，assert（）
{
    cutelog("wow I run to here now");
}

void ScatterBevPlugin::destroy() noexcept  // 老师说碰到的所有的destroy都是delete this
{
    cutelog("wow I run to here now");
    delete this;
}

void ScatterBevPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    cutelog("wow I run to here now");
    mNamespace = libNamespace;
}

const char* ScatterBevPlugin::getPluginNamespace() const noexcept
{
    cutelog("wow I run to here now");
    return mNamespace.c_str();
}

///////////////

ScatterBevPluginCreator::ScatterBevPluginCreator()  // 这边也是直接复制过来就行了，根本不用管
{
    cutelog("wow I run to here now");
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ScatterBevPluginCreator::getPluginName() const noexcept  // 跟上面的type是一样的，官方建议你设置都是一样的，那么这个区别是什么？
{
    cutelog("wow I run to here now");
    return PLUGIN_NAME;
}

const char* ScatterBevPluginCreator::getPluginVersion() const noexcept  //同name
{
    cutelog("wow I run to here now");
    return PLUGIN_VERSION;
}

const PluginFieldCollection* ScatterBevPluginCreator::getFieldNames() noexcept // 不用管，直接返回就行了
{
    cutelog("wow I run to here now");
    return &mFC;
}


// 这个，实际上也是调用了一个什么nre 构造函数。这个函数的参数是统一的。可以自己调用new。
// 需要把自己的plugin打造成对外使用的plugin，直接new的话，相当于把自己的代码构造告诉别人了，设计暴露给别人了，但是用这个封装的话，就是直接给别人一个这个层，只要把参数传对了就可以调用plugin了
IPluginV2* ScatterBevPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    cutelog("wow I run to here now");
    return new ScatterBevPlugin();
}


// 反序列化，也就是调用上面那个反序列化的构造函数了，new一下      
IPluginV2* ScatterBevPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    cutelog("wow I run to here now");
    return new ScatterBevPlugin(serialData, serialLength);
}

void ScatterBevPluginCreator::setPluginNamespace(const char* libNamespace) noexcept  // 后面两个也是固定的基本
{
    cutelog("wow I run to here now");
    mNamespace = libNamespace;
}

const char* ScatterBevPluginCreator::getPluginNamespace() const noexcept
{
    cutelog("wow I run to here now");
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(ScatterBevPluginCreator);
