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
    out_h=output_spatial_size(H,kernel_h,stride,padding)
    out_w=output_spatial_size(W,kernel_w,stride,padding)
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

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    weights=cache["weights"]
    x_shape = cache["x_shape"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]
    Cout = weights.shape[0]
    w_flat=weights.reshape(Cout,-1)#shape:(Cout,Cin*kernel_h*kernel_w)
    d_out=np.transpose(d_out,(0,2,3,1))
    d_out_row=d_out.reshape(-1,Cout)#d_out_row shape:(N*out_h*out_w, C_out)
    d_cols=d_out_row@w_flat
    dx=col2im(d_cols,x_shape,kernel_h,kernel_w,stride,padding)
    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    weights = cache["weights"]
    cols=cache["cols"]
    Cout = weights.shape[0]
    d_out_flat=np.transpose(d_out,(0,2,3,1)).reshape(-1,Cout)
    dw_flat=d_out_flat.T@cols
    dw=dw_flat.reshape(weights.shape)
    return dw

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    return np.sum(d_out, axis=(0,2,3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    return (conv2d_grad_input(d_out,cache),conv2d_grad_weights(d_out,cache),conv2d_grad_bias(d_out))

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    N,C,H,W=x.shape
    out_h=output_spatial_size(H,kernel,stride,0)
    out_w=output_spatial_size(W,kernel,stride,0)
    out=np.zeros((N,C,out_h,out_w))
    argmax=np.zeros((N,C,out_h,out_w),dtype=int)
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    window=x[n,c,i*stride:i*stride+kernel,j*stride:j*stride+kernel]
                    flat=window.reshape(-1)
                    out[n,c,i,j]=np.max(flat)
                    argmax[n,c,i,j]=np.argmax(flat)
    cache={
        'x_shape':x.shape,
        'argmax':argmax,
        'kernel':kernel,
        'stride':stride

    }
    return out,cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    kernel_window=np.zeros((kernel,kernel))
    kernel_flat=kernel_window.reshape(-1)
    kernel_flat[argmax_index]=grad_value
    return kernel_flat.reshape(kernel_window.shape)

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    x_shape=cache['x_shape']
    argmax=cache['argmax']
    kernel=cache['kernel']
    stride=cache['stride']


    N,C,H,W=x_shape
    out_h=output_spatial_size(H,kernel,stride,0)
    out_w=output_spatial_size(W,kernel,stride,0)
    
    dx_grad=np.zeros(x_shape)
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    dx_grad[n,
                    c,
                    i*stride:i*stride+kernel,
                    j*stride:j*stride+kernel]+=scatter_grad_window(d_out[n,c,i,j],argmax[n,c,i,j],kernel)
    return dx_grad

# Step 25 - relu_forward
def relu_forward(x):
    cache={
        'x':x
    }
    return np.maximum(0,x),cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    x=cache['x']
    return d_out*(x>0)

# Step 27 - flatten_forward
def flatten_forward(x):
    N,C,H,W=x.shape
    cache={
        'x_shape':x.shape
    }
    
    return (x.reshape(N,-1),cache)

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    x_shape=cache['x_shape']
    return d_out.reshape(x_shape)

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    y=x@weights+bias
    cache={
        'x':x,
        'weights':weights
    }
    return y,cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    weights=cache['weights']
    d_x=d_out@weights.T
    return d_x

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    d_w=x.T@dout
    return d_w

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    return np.sum(dout,axis=0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    x=cache['x']
    return (linear_grad_input(dout,cache),linear_grad_weights(x,dout),linear_grad_bias(dout))

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    probs=stable_softmax(logits)
    return abs(cross_entropy_loss(probs,y,eps=1e-12))

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    prob=stable_softmax((logits))
    N=logits.shape[0]
    dlogits=(prob-one_hot(y,logits.shape[1]))/N
    return dlogits+0.0

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    return param-grad*lr

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    return beta_one*m+(1-beta_one)*grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    g_sq=np.square(grad)
    return beta_two*v+(1-beta_two)*g_sq

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    return moment/(1-beta**t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    denom=np.sqrt(v_hat)+eps
    return param-lr*(m_hat/denom)

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    m=adam_update_m(m,grad,beta_one)
    v=adam_update_v(v,grad,beta_two)
    m_hat=adam_bias_correct(m,beta_one,t)
    v_hat=adam_bias_correct(v,beta_two,t)
    return (adam_param_step(param,m_hat,v_hat,lr,eps),m,v)

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    fan_in=in_channels*kernel_size*kernel_size
    weights=he_init((out_channels,in_channels,kernel_size,kernel_size),fan_in,seed)
    bias=init_zero_bias(out_channels)
    return {
        'W':weights,
        'b':bias
    }

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    fan_in=in_features
    return {
        'W':he_init((in_features,out_features),fan_in,seed),
        'b':init_zero_bias(out_features)
    }

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    kernel_size=5
    return {
        'conv1':init_conv_layer(6,in_channels,kernel_size,seed),
        'conv2':init_conv_layer(16,6,kernel_size,seed+1),
        'fc1':init_linear_layer(256,120,seed+2),
        'fc2':init_linear_layer(120,num_classes,seed+3)
    }

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    conv_out,conv_cache=conv2d_forward(x,W,b,stride,pad)
    relu_out,relu_cache=relu_forward(conv_out)
    pool_out,pool_cache=maxpool2d_forward(relu_out,pool_size,pool_size)
    cache={
        'conv_cache':conv_cache,
        'relu_cache':relu_cache,
        'pool_cache':pool_cache
    }
    return pool_out,cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    flat_out,flatten_cache=flatten_forward(x)
    lin_out,fc1_cache=linear_forward(flat_out,fc1['W'],fc1['b'])
    relu_out,relu_cache=relu_forward(lin_out)
    logits,fc2_cache=linear_forward(relu_out,fc2['W'],fc2['b'])
    cache={
        'flatten_cache':flatten_cache,
        'fc1_cache':fc1_cache,
        'relu_cache':relu_cache,
        'fc2_cache':fc2_cache
    }
    return logits,cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    conv1,block1=forward_conv_block(x,params['conv1']['W'],params['conv1']['b'],pool_size=2,stride=1,pad=0)
    conv2,block2=forward_conv_block(conv1,params['conv2']['W'],params['conv2']['b'],pool_size=2,stride=1,pad=0)
    logits,classifier=forward_classifier_block(conv2,params['fc1'],params['fc2'])
    cache={
        'block1':block1,
        'block2':block2,
        'classifier':classifier
    }
    return logits,cache

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    d_relu = maxpool2d_backward(
        dout,
        cache['pool_cache']
    )

    d_conv = relu_backward(
        d_relu,
        cache['relu_cache']
    )

    dx, dw, db = conv2d_backward(
        d_conv,
        cache['conv_cache']
    )

    return dx, dw, db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    d_relu,dw2,db2=linear_backward(dlogits,cache['fc2_cache'])
    d_fc1=relu_backward(d_relu,cache['relu_cache'])
    d_flat,dw1,db1=linear_backward(d_fc1,cache['fc1_cache'])
    dx=flatten_backward(d_flat,cache['flatten_cache'])
    return {
        'dx':dx,
        'fc1':{
            'dW':dw1,
            'db':db1
        },
        'fc2':{
            'dW':dw2,
            'db':db2
        }
    }

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):

    classifier_grads = backward_classifier_block(
        dlogits,
        caches['classifier']
    )

    dx2, dW2, db2 = backward_conv_block(
        classifier_grads['dx'],
        caches['block2']
    )

    dx1, dW1, db1 = backward_conv_block(
        dx2,
        caches['block1']
    )

    return {
        'conv1': {
            'dW': dW1,
            'db': db1
        },
        'conv2': {
            'dW': dW2,
            'db': db2
        },
        'fc1': {
            'dW': classifier_grads['fc1']['dW'],
            'db': classifier_grads['fc1']['db']
        },
        'fc2': {
            'dW': classifier_grads['fc2']['dW'],
            'db': classifier_grads['fc2']['db']
        }
    }

# Step 51 - lenet_predict
def lenet_predict(x, params):
    logits, _ = lenet_forward(x, params)
    return np.argmax(logits, axis=1)

# Step 52 - build_synthetic_image_dataset
def build_synthetic_image_dataset(
    num_samples,
    num_classes,
    image_size,
    in_channels=1,
    seed=0
):
    rng = np.random.default_rng(seed)
    y = rng.integers(
        0,
        num_classes,
        size=num_samples
    )
    x = rng.standard_normal(
        (
            num_samples,
            in_channels,
            image_size,
            image_size
        )
    )
    shift = y - (num_classes - 1) / 2
    x += shift[:, None, None, None]
    return x, y

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    np.random.seed(seed)
    return np.random.permutation(n)

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    n = len(x)
    idx = shuffle_indices(n, seed)
    n_test = int(n * test_fraction)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    return x_train, y_train, x_test, y_test

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
    idx = shuffle_indices(len(x), seed)

    for start in range(0, len(idx) - batch_size + 1, batch_size):
        batch_idx = idx[start:start + batch_size]

        xb = x[batch_idx]
        yb = y[batch_idx]

        yield xb, yb

# Step 56 - train_step
def train_step(params, opt_state, xb, yb,
               lr, beta_one, beta_two, eps, step):

    logits, caches = lenet_forward(xb, params)

    loss = softmax_cross_entropy_forward(logits, yb)

    dlogits = softmax_cross_entropy_backward(logits, yb)

    grads = lenet_backward(dlogits, caches)

    new_params = {}
    new_opt_state = {}

    for layer in ['conv1', 'conv2', 'fc1', 'fc2']:

        new_params[layer] = {}
        new_opt_state[layer] = {}

        W, mW, vW = adam_step(
            params[layer]['W'],
            grads[layer]['dW'],
            opt_state[layer]['W']['m'],
            opt_state[layer]['W']['v'],
            step,
            lr,
            beta_one,
            beta_two,
            eps
        )

        b, mb, vb = adam_step(
            params[layer]['b'],
            grads[layer]['db'],
            opt_state[layer]['b']['m'],
            opt_state[layer]['b']['v'],
            step,
            lr,
            beta_one,
            beta_two,
            eps
        )

        new_params[layer]['W'] = W
        new_params[layer]['b'] = b

        new_opt_state[layer]['W'] = {
            'm': mW,
            'v': vW
        }

        new_opt_state[layer]['b'] = {
            'm': mb,
            'v': vb
        }

    return new_params, new_opt_state, loss

# Step 57 - train_one_epoch
def train_one_epoch(params, opt_state, x, y,
                    batch_size, lr,
                    beta_one, beta_two,
                    eps, step_counter,
                    seed=0):

    losses = []

    for xb, yb in iterate_minibatches(
        x, y, batch_size, seed
    ):

        step_counter += 1

        params, opt_state, loss = train_step(
            params,
            opt_state,
            xb,
            yb,
            lr,
            beta_one,
            beta_two,
            eps,
            step_counter
        )

        losses.append(loss)

    return params, opt_state, step_counter, losses

# Step 58 - train_loop
def train_loop(params, x_train, y_train,
               num_epochs, batch_size,
               lr=1e-3,
               beta_one=0.9,
               beta_two=0.999,
               eps=1e-8,
               seed=0):

    opt_state = {}

    for layer in ['conv1', 'conv2', 'fc1', 'fc2']:

        opt_state[layer] = {}

        for pname in ['W', 'b']:

            param = params[layer][pname]

            opt_state[layer][pname] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param)
            }

    step_counter = 0
    loss_history = []

    for epoch in range(num_epochs):

        params, opt_state, step_counter, losses = train_one_epoch(
            params,
            opt_state,
            x_train,
            y_train,
            batch_size,
            lr,
            beta_one,
            beta_two,
            eps,
            step_counter,
            seed + epoch
        )

        loss_history.extend(losses)

    return params, loss_history

# Step 59 - evaluate
def evaluate(params, x, y):
    pred = lenet_predict(x, params)
    return float(np.mean(pred == y))

