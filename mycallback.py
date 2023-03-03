import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class PlotImages(tf.keras.callbacks.Callback):
    """
     This class generates, plots and saves the fake images from the generator model during training.
    """
    def __init__(self, noise_dim, batch_size, save_epoch_results, show_epoch_results, patience_to_show, nb_rows, nb_cols, **kwargs):
        """
        Initialize the class.
    
        Args:
        - noise_dim (int): dimension of the noise vector to generate fake images.
        - batch_size (int): number of images to be generated in a single batch.
        - save_epoch_results (bool): boolean indicating whether the results of the current epoch should be saved or not.
        - show_epoch_results (bool): boolean for showing generated images every epoch.
        - patience_to_show (int): number of epochs to wait before showing the generated images, if 'show_epoch_results' == True.
        - nb_rows (int): number of rows in the grid of images to plot.
        - nb_cols (int): number of columns in the grid of images to plot.
        """
        super(tf.keras.callbacks.Callback, self).__init__(**kwargs)
        self.noise = tf.random.normal([batch_size, noise_dim])
        self.save_epoch_results = save_epoch_results
        self.show_epoch_results = show_epoch_results
        self.patience_to_show = patience_to_show
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols

    def on_epoch_end(self, epoch, logs=None):
        """
        This function generates, plots and saves images after all epoch.

        Args:
        - epoch (int): current epoch.
        - logs (dict): epoch logs.
        """
        # Predict fake images
        fake_images = self.model.generator.predict(self.noise)

        # Normalize the fake images to range [0,1] for visualization
        fake_images = (fake_images+1)/2
        # Clip the values to range [0,1]
        fake_images = np.array(tf.clip_by_value(fake_images, clip_value_min=0., clip_value_max=1.))

        # Check if the fake images have only one channel (grayscale)
        if fake_images.shape[-1] == 1:
            # Remove the redundant dimension if grayscale
            fake_images = np.squeeze(fake_images)

        # Loop over number of samples to be plotted
        for i in range(self.nb_rows*self.nb_cols):
            # Plot each image in a subplot
            plt.subplot(self.nb_rows, self.nb_cols, i+1)
            plt.imshow(fake_images[i])
            plt.axis("off")

        # Save the generated fake images if 'save_epoch_results' = True
        if self.save_epoch_results:
            # Save the images in jpg format
            if os.path.exists("./epoch_outputs") == False:
                os.makedirs("./epoch_outputs")
                
            plt.savefig(f"./epoch_outputs/epoch{epoch+1}.jpg")
            # Print the path of the saved image
            print(f"Epoch {epoch+1} results saved to ./epoch_outputs/epoch{epoch+1}.jpg")

        # Plot the generated fake images if 'show_epoch_results' = True 
        if self.show_epoch_results:    
            if (epoch+1) % self.patience_to_show == 0:
                plt.show()

