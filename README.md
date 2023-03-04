# Anime Face Generation using GAN Model [Tensorflow]
A Generative Adversarial Network (GAN) is a deep learning model used for generating new data samples from an existing dataset. The model consists of two neural networks, the generator and the discriminator, which are trained simultaneously in a zero-sum game manner. The generator aims to generate data samples that are indistinguishable from the real ones, while the discriminator tries to identify which samples are real and which ones are generated. The generator and the discriminator continuously improve until the generator produces realistic data samples that the discriminator can no longer distinguish. 

![GANModel](https://user-images.githubusercontent.com/99184963/222919024-04936461-fb2c-478b-a240-1b9251b97d1e.png)
>Retrieved from [Google Developers](https://developers.google.com/machine-learning/gan/generator) 

In this repository, the GAN model has been trained on  anime face dataset to generate new synthetic anime faces.

**NOTE: You can use the repository not only for this dataset, but also for any dataset you want. You are very free to adjust the hyperparameters of the GAN model to improve your model for your project.**

## Dataset
This dataset which the GAN Model has been trained is consisting of 63,632 anime faces. It contains high quality anime character images with clean background and rich colors. 

The dataset is not included in this repository, but you can download it from this [link](https://www.kaggle.com/datasets/splcher/animefacedataset)

## Usage

### ***Cloning***

>To use the GAN model in your projects like generating anime faces, clone this repository using your terminal like following command;

    git clone https://github.com/UluLord/Anime-Face-Generation-using-GAN-Model-Tensorflow-.git

> After cloning, change the directory you are working to the repository directory;

    cd Anime-Face-Generation-using-GAN-Model-Tensorflow-

### ***Requirements***

This work has been tested on these libraries;

* Tensorflow: 2.11.0
* Numpy: 1.24.1
* Matplotlib: 3.6.3

>To install the required packages, run the following command;

    pip install -r requirements.txt

**NOTE: It may work with other versions of the libraries, but this has not been tested.**

* This work has been tested on NVIDIA GeForce RTX 3060 GPU.

**NOTE: It is highly recommended to work with a GPU.**
    
### ***Training GAN Model***

>Then, use the **train.py** with desired arguments to train the model on your dataset.

* Adjust the parameters if necessary;
  * **dataset_path**: Path to image dataset. Required.
  * **height**: Height of image. Default is 64.
  * **width**: Width of image. Default is 64.
  * **nb_channels**: Number of channels of images (choose one of those: 1 or 3). Default is 3.
  * **epochs**: Number of training epochs. Default is 100.
  * **noise_dim** Dimension of the noise vector to generate fake images. Default is 100.
  * **batch_size**: The batch size for dividing the dataset into chunks. Default is 64.
  * **buffer_size**: Buffer size for shuffling the dataset. Default is 1000.
  * **nb_rows**: Number of rows in the grid of images to plot. Default is 5.
  * **nb_cols**: Number of columns in the grid of images to plot. Default is 5.
  * **save_epoch_results**: Option to save epoch results. Default is True.
  * **show_epoch_results**: Option to show epoch results. Default is False.
  * **patience_to_show**: Number of epochs to wait before showing the generated images, if 'show_epoch_results' is True. Default is 1.
  * **generator_optimizer**: Generator model optimizer (choose one of those: sgd, rmsprop or adam). Default is ‘adam’.
  * **discriminator_optimizer**: Discriminator model optimizer (choose one of those: sgd, rmsprop or adam). Default is ‘adam’.
  * **learning_rate**: The learning rate used during training. Default is 0.001.
  * **beta_1**: The first hyperparameter for the Adam optimizer. Default is 0.9.
  * **beta_2**: The second hyperparameter for the Adam optimizer. Default is 0.999.
  * **epsilon**: A small constant added to the denominator to prevent division by zero. Default is 1e-7.
  * **momentum**: Momentum term for the SGD optimizer. Default is 0.
  * **nesterov**: Whether to use Nesterov momentum for the SGD optimizer. Default is False.
  * **rho**: Decay rate for the moving average of the squared gradient for the RMSprop optimizer. Default is 0.9
  * **rmsprop_momentum**: The momentum term for the RMSprop optimizer. Default is 0.

* Example usage;

> Write the code like following command;

    python train.py --dataset_path ./image-dataset/ --height 64 --width 64 --nb_channels 3 --epochs 150 --noise_dim 100 --batch_size 128

* Sample Training Process;

Following are images, generated by the Generator Model during training;

![training_process](https://user-images.githubusercontent.com/99184963/222831324-c0d5802b-2bfc-4c27-8a0c-43b1055140df.gif)

**NOTE: It is possible to get better results by adjusting the parameters, like using lower learning rate or different optimizer type.**

### ***Generate New Anime Faces***

> To generate new images by using a pre-trained model;

* Specify a pre-trained model (download new one or train a model by using train.py).
* Use **generate.py** with desired arguments.

* Adjust the parameters if necessary;
    
    * **pretrained_model_path**: Path to the pretrained generator model. Default is ‘./pretrained_models/generator_model’.
    * **nb_rows**: Number of rows in the grid of images to plot. Default is 5.
    * **nb_cols**: Number of columns in the grid of images to plot. Default is 5.
    * **show_generated**: Option to show generated images. Default is True.
    * **save_to_dir**: Option to save generated images to a directory. Default is ‘./generated_by_model/generated.jpg’
    * **noise_dim**: Dimensionality of the noise vector to be fed to the generator. Default is 100. ***(NOTE: specify noise_dim in harmony with model)***
    * **batch_size**: Number of images to be generated in a single batch. Default is 64. ***(NOTE: specify batch_size in harmony with model)***

* Example usage;

 > Write the code like following command;

    python generate.py --pretrained_model_path ./pretrained_models/generator_model --nb_rows 5 --nb_cols 5 --noise_dim 100 --batch_size 128 --show_generated True --save_to_dir True


* Sample Generated Images;

Following are some of the images, generated by the Generator Model after the model trained: 

![generated](https://user-images.githubusercontent.com/99184963/222833092-a94af223-70db-47de-b9fd-13a1e37735e9.jpg)

## Citation

    @online{chao2019/online,
  	        author = {Brian Chao},
  	        title = {Anime Face Dataset: a collection of high-quality anime faces.},
  	        date = {2019-09-16},
  	        url = {https://github.com/bchao1/Anime-Face-Dataset}
            }

    @misc{spencer churchill_brian chao_2019,
	      title={Anime Face Dataset},
	      url={https://www.kaggle.com/ds/379764},
	      DOI={10.34740/KAGGLE/DS/379764}
          }

> If you use this repository in your work, please consider citing us as the following.

    @misc{ululord2023anime-face-generation-using-gan-model-tensorflow,
	      author = {Fatih Demir},
          title = {Anime Face Generation using GAN Model [Tensorflow]},
          date = {2023-03-04},
          url = {https://github.com/UluLord/Anime-Face-Generation-using-GAN-Model-Tensorflow-}
          }
