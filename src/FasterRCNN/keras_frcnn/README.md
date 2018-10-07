This implementation of Faster RCNN is based on the code of [this](https://github.com/yhenon/keras-frcnn/tree/0528b81ae15ee861f5d01ec6c758554b30842fff) repo.

You have to download the following files in the pretrained-models folder:

- resnet50_weights_tf_dim_ordering_tf_kernels.h5
- vgg16_weights_tf_dim_ordering_tf_kernels.h5

You can find the pre trained models [here](https://github.com/fchollet/deep-learning-models/releases).

To start training run:


`python train_frcnn.py -o rsna -p ..\..\input\`

`..\..\input` is the folder with the training data. This folder must contain:

- `stage_1_train_labels.csv`
- A folder called `stage_1_train_images`, which contains all the dcm files.

Some relevant arguments to pass when training:

- `--network` to pick the VGG pretrained model (default is resnet). 
- `--num_rois` number of regions of interested to process at once.
- `--num_epochs` self explanatory.

Check `train_frcnn.py` for details on all the arguments.

Check `config.py` for some default values and another parameters during training. For example, `im_size` defined there determine the input size of the image (600x600).

The model gets saved to `model_frcnn.vgg.hdf5` whenever the loss is improved.

**TODO**:

- Transform this  to a TPU estimator (see [this](https://github.com/tensorflow/tpu/blob/master/models/experimental/cifar_keras/cifar_keras.py) example). For this we'll probably need to do (this is a draft, could be less or more):
  1. Convert all the losses to tensorflow ops.
  2. Get the optimizer op on every iteration. 
  3. Create a TPU estimator which returns a `TPUEstimatorSpec`.
- Add more pre trained models: in the [keras repo](https://github.com/fchollet/deep-learning-models/releases) there are some interesting models such as dense net. We'll have to write the architecture as in `resnet.py` and `vgg.py`. Of course, this does not have to be restricted to keras pretrained models, any keras model (backed by tf) should work. Specially [Feature Pyramid networks](https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c).
- Check `test_frcnn` and add the RSNA challenge metrics.

 