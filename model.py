"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np
def argmax_rows(matrix):
    return np.argmax(matrix,axis=1)
    # TODO: return the index of the largest element in each row of a 2D array

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    return np.max(matrix,axis=1,keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    return np.sum(matrix,axis=1,keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    return np.exp(logits-row_max(logits))

# Step 5 - stable_softmax
def stable_softmax(logits):
    return exp_shifted(logits)/row_sum(exp_shifted(logits))

# Step 6 - one_hot
def one_hot(labels, num_classes):
    hot_encoded=np.zeros((len(labels),num_classes),dtype=float)
    rows=np.arange(len(labels))
    hot_encoded[rows,labels]=1
    return hot_encoded

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    rows=np.arange(len(labels))
    return probs[rows,labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    return -np.mean(np.log(np.maximum(gather_true_class_probs(probs,labels),eps)))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    predicted=argmax_rows(logits_or_probs)
    return np.mean((predicted==labels))

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2.0/fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    np.random.seed(seed)
    return np.random.normal(
        loc=0.0,
        scale=he_std(fan_in),
        size=shape).astype(np.float64)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    return np.zeros(length)

# Step 13 - pad_2d
def pad_2d(images, pad):
    return np.pad(images,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant', constant_values=0)

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    return ((input_size+2*padding-kernel)//stride)+1

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    images=pad_2d(images,padding)
    N,C,H,W=images.shape
    out_h=output_spatial_size(H,kernel_h,stride,0)
    out_w=output_spatial_size(W,kernel_w,stride,0)
    imcols=np.zeros((N*out_h*out_w,C*kernel_h*kernel_w),dtype=images.dtype)
    rows=0
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                patch=images[n,:,i*stride:i*stride+kernel_h,j*stride:j*stride+kernel_w]
                imcols[rows]=patch.reshape(-1)
                rows+=1
    return imcols

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N,C,H,W=input_shape
    H_pad=H+2*padding
    W_pad=W+2*padding
    out_h=output_spatial_size(H_pad,kernel_h,stride,0)
    out_w=output_spatial_size(W_pad,kernel_w,stride,0)
    images=np.zeros((N,C,H_pad,W_pad))
    rows=0
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                patch=cols[rows].reshape(C,kernel_h,kernel_w)
                images[n,:,i*stride:i*stride+kernel_h,j*stride:j*stride+kernel_w]+=patch
                rows+=1
    
    if padding>0:
        images=images[:,:,padding:-padding,padding:-padding]
    return images

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
    N,Cin,H,W=x.shape
    Cout,Cin,kernel_h,kernel_w=weights.shape
    Xcols=im2col(x,kernel_h,kernel_w,stride,padding)
    out_h=output_spatial_size(H,kernel_h,stride,0)
    out_w=output_spatial_size(W,kernel_w,stride,0)
    w_flat=weights.reshape(Cout,-1)
    w_flat_transpose=np.transpose(w_flat,(1,0)) #or shortcut we can write w_flat.T
    y=Xcols@w_flat_transpose+bias
    y=y.reshape(N,out_h,out_w,Cout)
    y=np.transpose(y,(0,3,1,2))
    cache_dic={
        'x_shape':x.shape,
        'weights':weights,
        'cols':Xcols,
        'stride':stride,
        'padding':padding,
        'kernel_h':kernel_h,
        'kernel_w':kernel_w
    }
    return y,cache_dic

# Step 18 - conv2d_grad_input (not yet solved)
# TODO: implement

# Step 19 - conv2d_grad_weights (not yet solved)
# TODO: implement

# Step 20 - conv2d_grad_bias (not yet solved)
# TODO: implement

# Step 21 - conv2d_backward (not yet solved)
# TODO: implement

# Step 22 - maxpool2d_forward (not yet solved)
# TODO: implement

# Step 23 - scatter_grad_window (not yet solved)
# TODO: implement

# Step 24 - maxpool2d_backward (not yet solved)
# TODO: implement

# Step 25 - relu_forward (not yet solved)
# TODO: implement

# Step 26 - relu_backward (not yet solved)
# TODO: implement

# Step 27 - flatten_forward (not yet solved)
# TODO: implement

# Step 28 - flatten_backward (not yet solved)
# TODO: implement

# Step 29 - linear_forward (not yet solved)
# TODO: implement

# Step 30 - linear_grad_input (not yet solved)
# TODO: implement

# Step 31 - linear_grad_weights (not yet solved)
# TODO: implement

# Step 32 - linear_grad_bias (not yet solved)
# TODO: implement

# Step 33 - linear_backward (not yet solved)
# TODO: implement

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

