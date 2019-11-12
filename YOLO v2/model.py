# 使用基于 tensorflow的keras 的 Sequential 顺序模型来实现yolo v2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Softmax, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2


# 实例化Sequential模型
model = Sequential()
# 第一层 卷积层 Convolutional 卷积核 3*3 步长 1 filters 32 outpot 224*224
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第二层 进行最大池化 size=(2, 2) strides = 2 outpot 112*112
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=2,
    padding='same',
    data_format=None
))
#第三层 Convolutional filters=64, size=(3, 3) strides=1, output 112*112
model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第四层 Maxpool size=(2, 2) strides = 2 outpot 56*56
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=2,
    padding='same',
    data_format=None
))
# 第五层 Convolutional filters=128, size=(3, 3) strides=1, output 56*56
model.add(Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第六层 Convolutional filter64, size=(1, 1) strides=1, output=56
model.add(Conv2D(
    filters=64,
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第七层 Convolutional filter 128, size=(3, 3) strides=1, output=56*56
model.add(Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第八层 Maxpool size=(2, 2) strides = 2 outpot 28*28
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=2,
    padding='same',
    data_format=None
))
# 第九层 Convolutional filter 256, size=(3, 3) strides=1, output=28*28
model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第十层 Convolutional filter 128 size=(1, 1) strides=1, output=28*28
model.add(Conv2D(
    filters=128,
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第十一层 Convolutional filter 256 size=(3, 3) strides=1, output=28*28
model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第十二层 Maxpool size=(2, 2) strides = 2 outpot 14*14
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=2,
    padding='same',
    data_format=None
))
# 第十三层 Convolutional filter 512 size=(3, 3) strides=1, output=14*14
model.add(Conv2(
    filters=512,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第十四层 Convolutional filter 256 size=(1, 1) strides=1, output=14*14
model.add(Conv2(
    filters=256,
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第十五层 Convolutional filter 512 size=(3, 3) strides=1, output=14*14
model.add(Conv2(
    filters=512,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第十六层 Convolutional filter 256 size=(1, 1) strides=1, output=14*14
model.add(Conv2(
    filters=256,
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第十七层 Convolutional filter 512 size=(3, 3) strides=1, output=14*14
model.add(Conv2(
    filters=512,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第十八层 Maxpool size=(2, 2) strides = 2 outpot 7*7
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=2,
    padding='same',
    data_format=None
))
# 第十九层 Convolutional filter 1024 size=(3, 3) strides=1, output=7*7
model.add(Conv2(
    filters=1024,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第二十层 Convolutional filter 512 size=(1, 1) strides=1, output=7*7
model.add(Conv2(
    filters=512,
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第二十一层 Convolutional filter 1024 size=(3, 3) strides=1, output=7*7
model.add(Conv2(
    filters=1024,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第二十二层 Convolutional filter 512 size=(1, 1) strides=1, output=7*7
model.add(Conv2(
    filters=512,
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第二十三层 Convolutional filter 1024 size=(3, 3) strides=1, output=7*7
model.add(Conv2(
    filters=1024,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
# 第二十四层 Convolutional filter 1024 size=(3, 3) strides=1, output=7*7
model.add(Conv2(
    filters=1000,
    kernel_size=(1, 1),
    strides=(1, 1),
    padding='same',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=l2(0.0005),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
))
# 进行batch normalization
model.add(BatchNormalization(
    axis=-1,
    momentum=0.90,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    renorm=False,
    renorm_clipping=None,
    renorm_momentum=0.99,
    fused=None,
    trainable=True,
    virtual_batch_size=None,
    adjustment=None,
    name=None,
    **kwargs
))
model.add(LeakyReLU(
    alpha = 0.1
))
model.add(GlobalAveragePooling2D())