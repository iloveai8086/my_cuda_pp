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

#include <iostream>
#include <sstream>
#include <fstream>

#include "cuda_runtime.h"

#include "./params.h"
#include "./pointpillar.h"

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

std::string Data_File = "../data4/";
std::string Model_File = "/home/ros/CLionProjects/CUDA-PointPillars2/model/pointpillar.onnx_best4";

void Getinfo(void)
{
  cudaDeviceProp prop;
//  std::cout<<"1111"<<std::endl;
  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);  // 调用read方法

  if (!dataFile.is_open())
  {
	  std::cout << "Can't open files: "<< file<<std::endl;
	  return -1;
  }

  //get length of file:
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end); //到结尾
  len = dataFile.tellg();  // 得到长度
  dataFile.seekg (0, dataFile.beg);  // 回到开头么？

  //allocate memory:
  char *buffer = new char[len];
  if(buffer==NULL) {
	  std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
	  exit(-1);
  }

  //read data as a block:
  dataFile.read(buffer, len);  // 读取进来二进制文件
  dataFile.close();  // 关闭

  *data = (void*)buffer; // 传进来的是两个地址么，我看的是&
  *length = len;
  return 0;  
}

int main(int argc, const char **argv)
{
  Getinfo();

  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaStream_t stream = NULL;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  Params params_;

  std::vector<Bndbox> nms_pred;
  nms_pred.reserve(100);

  PointPillar pointpillar(Model_File, stream);  // 分配了一堆内存	

  for (int i = 0; i < 10; i++)
  {
    std::string dataFile = Data_File;

    std::stringstream ss;

    ss<< i;  // 赋值给ss

    dataFile +="00000";
    dataFile += ss.str();
    dataFile +=".bin";
//    dataFile += "00000.bin";
    std::cout << "<<<<<<<<<<<" <<std::endl;
    std::cout << "load file: "<< dataFile <<std::endl;

    //load points cloud
    unsigned int length = 0;
    void *data = NULL;
    std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>()); // 智能指针
    loadData(dataFile.data(), &data, &length);
    buffer.reset((char *)data);  // 函数的作用是将引用计数减1，停止对指针的共享，除非引用计数为0，否则不会发生删除操作

    float* points = (float*)buffer.get();
    size_t points_size = length/sizeof(float)/4; // 605840 / 4 / 4   605840 是二进制文件总共占得字节

    std::cout << "find points num: "<< points_size <<std::endl;

    float *points_data = nullptr;
    unsigned int points_data_size = points_size * 4 * sizeof(float);  // 就是length
    checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size)); // 先做个内存的分配 把之前定义的空指针先分配了内存，和malloc普通的有啥区别
    checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault)); // 这边就是把内存直接拷贝过去，内存地址，大小，方式
    checkCudaErrors(cudaDeviceSynchronize()); // 同步

    cudaEventRecord(start, stream);

    pointpillar.doinfer(points_data, points_size, nms_pred);  //传入分配的内存（显存），此时有数据了，点的个数，nmspred数据结构
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout<<"TIME: pointpillar: "<< elapsedTime <<" ms." <<std::endl;

    checkCudaErrors(cudaFree(points_data));

    std::cout<<"Bndbox objs: "<< nms_pred.size()<<std::endl;

	for (auto i = nms_pred.begin(); i != nms_pred.end(); i++) {
            std::cout << i->x << ' ' << i->y << ' ' << i->z << ' ' << i->l << ' ' << i->w << ' ' << i->h << ' ' << i->score
                 << ' ' << i->rt << ' ' << i->id << std::endl;}

    nms_pred.clear();
    std::cout << ">>>>>>>>>>>" <<std::endl;
  }

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));

  return 0;
}
