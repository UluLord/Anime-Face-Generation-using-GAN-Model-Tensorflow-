import os
import tensorflow as tf

class GetDataset:
    """
    This class can be used to load, preprocess, and create a TensorFlow dataset of images.
    """

    def __init__(self, height, width, nb_channels, batch_size, buffer_size):
        """
        Initialize the class.

        Args:
        - height (int): height of image.
        - width (int): width of image.
        - nb_channels (int): number of channels of images in dataset.
        - batch_size (int): batch size for dividing the dataset into chunks.
        - buffer_size (int): buffer size for shuffling the dataset.
        """
        self.nb_channels = nb_channels
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def list_dataset(self, image_dataset_path):
        """
        This function lists all image dataset and converts it into a tensor list.

        Args:
        - image_dataset_path (str): path to the image dataset directory.

        Returns:
        - tf_dataset (Tensorflow dataset): Tensorflow dataset including the images.
        """
        # List image dataset
        image_list = os.listdir(image_dataset_path)
        image_ds_list = [image_dataset_path + "/" + i for i in image_list]

        # Sort the list
        image_ds_list = sorted(image_ds_list)

       # Print dataset size
        print("There are totally {} images in dataset".format(len(image_ds_list)))

        # Converts the list into a tensor list
        tf_dataset = tf.data.Dataset.from_tensor_slices(image_ds_list)

        return tf_dataset

    def preprocessing(self, image_dataset_path):
        """
        This function preprocesses the image dataset.

        Args:
        - image_dataset_path (tensor): path to the image dataset.

        Returns:
        - images (tensor): preprocessed image dataset.
        """
        # Preprocess image dataset
        images = tf.io.read_file(image_dataset_path)
        images = tf.image.decode_png(images, channels=self.nb_channels)
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.image.resize(images, size=(self.height, self.width), method="nearest")
        images = images * 2 - 1

        return images

    def __call__(self, image_dataset_path):
        """
        Calling this class applies the preprocessing function to each element in the dataset, 
        shuffles the dataset, batches the dataset, and prefetches the data.

        Args:
        - image_dataset_path (str): path to the image dataset directory.

        Returns:
        - dataset (Tensorflow dataset): Tensorflow dataset to use for training.
        """
        tf_dataset = self.list_dataset(image_dataset_path)
        
        dataset = tf_dataset.map(self.preprocessing)
        
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset