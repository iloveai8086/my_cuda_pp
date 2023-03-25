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
#include <cuda_runtime_api.h>
#include <stdio.h>

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

__device__ float sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void postprocess_kernal(const float *cls_input,
                                   float *box_input,
                                   const float *dir_cls_input,
                                   float *anchors,
                                   float *anchor_bottom_heights,
                                   float *bndbox_output,
                                   int *object_counter,
                                   const float min_x_range,
                                   const float max_x_range,
                                   const float min_y_range,
                                   const float max_y_range,
                                   const int feature_x_size,
                                   const int feature_y_size,
                                   const int num_anchors,
                                   const int num_classes,
                                   const int num_box_values,
                                   const float score_thresh,
                                   const float dir_offset) {
    // printf("we are in the post process kernel! \n");
    int loc_index = blockIdx.x;
    int ith_anchor = threadIdx.x;
    if (ith_anchor >= num_anchors) {
        return;
    }
    //printf("post process feature_x_size:%d",feature_x_size);
    int col = loc_index % feature_x_size;
    int row = loc_index / feature_x_size;
    float x_offset = min_x_range + col * (max_x_range - min_x_range) / (feature_x_size - 1);
    float y_offset = min_y_range + row * (max_y_range - min_y_range) / (feature_y_size - 1);
    //printf("%f\n",x_offset);   这边也是正常的
    int cls_offset = loc_index * num_anchors * num_classes + ith_anchor * num_classes;
    float dev_cls[2] = {-1, 0};

    const float *scores = cls_input + cls_offset;
    float max_score = sigmoid(scores[0]);
    int cls_id = 0;
    for (int i = 1; i < num_classes; i++) {  // 选出哪一个类别的得分最大
        float cls_score = sigmoid(scores[i]);
        if (cls_score > max_score) {
            max_score = cls_score;
            cls_id = i;
        }
    }
    dev_cls[0] = static_cast<float>(cls_id);  //  class-id
    dev_cls[1] = max_score;  // class score

//    if(dev_cls[1] < score_thresh){
//        //printf("we are in the post process kernel! and the score_thresh is %f \n",score_thresh);
//        bndbox_output[0] = 0;
//    }

    if (dev_cls[1] >= score_thresh) {
        //printf("we are in the post process kernel! and the score_thresh is %f \n",score_thresh);
        int box_offset = loc_index * num_anchors * num_box_values + ith_anchor * num_box_values;
        // num_box_values = 7
        int dir_cls_offset = loc_index * num_anchors * 2 + ith_anchor * 2;
        float *anchor_ptr = anchors + ith_anchor * 4;
        float z_offset = anchor_ptr[2] / 2 + anchor_bottom_heights[ith_anchor / 2];
        float anchor[7] = {x_offset, y_offset, z_offset, anchor_ptr[0], anchor_ptr[1], anchor_ptr[2], anchor_ptr[3]};
        float *box_encodings = box_input + box_offset;

        float xa = anchor[0];
        float ya = anchor[1];
        float za = anchor[2];
        float dxa = anchor[3];
        float dya = anchor[4];
        float dza = anchor[5];
        float ra = anchor[6];
        float diagonal = sqrtf(dxa * dxa + dya * dya);
        box_encodings[0] = box_encodings[0] * diagonal + xa;
        box_encodings[1] = box_encodings[1] * diagonal + ya;
        box_encodings[2] = box_encodings[2] * dza + za;
        box_encodings[3] = expf(box_encodings[3]) * dxa;
        box_encodings[4] = expf(box_encodings[4]) * dya;
        box_encodings[5] = expf(box_encodings[5]) * dza;
        box_encodings[6] = box_encodings[6] + ra;

        float yaw;
        int dir_label = dir_cls_input[dir_cls_offset] > dir_cls_input[dir_cls_offset + 1] ? 0 : 1;
        float period = 2 * M_PI / 2;
        float val = box_input[box_offset + 6] - dir_offset;
        float dir_rot = val - floor(val / (period + 1e-8) + 0.f) * period;
        yaw = dir_rot + dir_offset + period * dir_label;

        int resCount = (int) atomicAdd(object_counter, 1);
        //这个地方就是需要注意原子操作，是为了防止多线程的时候写错了，就是在排队
        // printf("the res count is %d ~~~~~~~~~~~~~~~~~~~~~\n",resCount);
        bndbox_output[0] = resCount + 1;
        // 但是下面的这个操作不是原子操作，所以会导致有问题，可能极端的概率下覆盖写入导致结果的错误
        // 一个可行的方法就是，用空间换时间的写法，就是有多少分配多少，避免原子操作的使用
        // 比如给每一个grid都分配一个内存，现在是9个分配成十个之类的操作
        float *data = bndbox_output + 1 + resCount * 9;
        data[0] = box_input[box_offset];
        data[1] = box_input[box_offset + 1];
        data[2] = box_input[box_offset + 2];
        data[3] = box_input[box_offset + 3];
        data[4] = box_input[box_offset + 4];
        data[5] = box_input[box_offset + 5];
        data[6] = yaw;
        data[7] = dev_cls[0];
        data[8] = dev_cls[1];
    }

}

cudaError_t postprocess_launch(const float *cls_input,
                               float *box_input,
                               const float *dir_cls_input,
                               float *anchors,
                               float *anchor_bottom_heights,
                               float *bndbox_output,
                               int *object_counter,
                               const float min_x_range,
                               const float max_x_range,
                               const float min_y_range,
                               const float max_y_range,
                               const int feature_x_size,
                               const int feature_y_size,
                               const int num_anchors,
                               const int num_classes,
                               const int num_box_values,
                               const float score_thresh,
                               const float dir_offset,
                               cudaStream_t stream) {

    int bev_size = feature_x_size * feature_y_size;  // 这个bev的size是干啥的
    dim3 threads(num_anchors);
    dim3 blocks(bev_size);
    std::cout << "post process feature size X:" << feature_x_size << std::endl;
    std::cout << "post process feature size Y:" << feature_y_size << std::endl;
    bndbox_output[0]=0;

    postprocess_kernal<<<blocks, threads, 0, stream>>>
            (cls_input,
             box_input,
             dir_cls_input,
             anchors,
             anchor_bottom_heights,
             bndbox_output,
             object_counter,
             min_x_range,
             max_x_range,
             min_y_range,
             max_y_range,
             feature_x_size,
             feature_y_size,
             num_anchors,
             num_classes,
             num_box_values,
             score_thresh,
             dir_offset);

    return cudaGetLastError();
}

