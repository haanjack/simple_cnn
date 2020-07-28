# simple conv2d network

Benchmark Conv2D layers with different weight sizes.

## How to profile?

```bash
nsys profile -y 60 -d 60 -o cnn_benchmark -t osrt,cuda,cublas,cudnn,nvtx python mnist_cnn.py
```

This command generates `cnn_benchmark.qdrep` file. You may want download that file and open with [Nsight System](https://developer.nvidia.com/nsight-systems). If you have install CUDA toolkit on your host machine, you don't have to install this manually.

## What to profile?

You can measure your desired convolutional layer's elapsed time with the different weight size. The following codes shows the sample test space.

```python
for kernelSize_y in [4, 16, 64, 256]:
    for kernelSize_x in [2, 7, 12, 17]:
        message='Conv2D (%d, %d)' % (kernelSize_x, kernelSize_y)
        print(message)
        x, marker_id, domain_id = NVTXStart(message=message, domain_name='forward', trainable=True)(x)
        x = Conv2D(filters=16, kernel_size=(kernelSize_x, kernelSize_y), activation='relu', padding='same')(x)
        x = NVTXEnd(grad_message=message, grad_domain_name='backwards')([x, marker_id, domain_id])
```

