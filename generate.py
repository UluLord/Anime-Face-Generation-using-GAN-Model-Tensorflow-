import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def set_generate_arguments():
    """
    This function parses command line arguments and returns them as a dictionary. 
    """
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(
        # Description of the project
        description="This project aims to implementation of a Generative Adversarial Network (GAN) for generating anime faces. \n\nTo train the model or to generate new anime faces, adjust the parameters if necessary:",
        # Usage string to display
        usage="Generating Anime Faces using GAN Model",
        # Set the formatter class to ArgumentDefaultsHelpFormatter
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # Add help argument
        add_help=True,
        # Set prefix chars
        prefix_chars="-",
        # Set default value for argument_default
        argument_default=argparse.SUPPRESS,
        # Allow abbreviations of options
        allow_abbrev=True
    )

    #Add arguments
    parser.add_argument("--pretrained_model_path", default="./pretrained_models/generator_model", type=str, required=False, 
                        help="Path to the pretrained generator model")
    parser.add_argument("--nb_rows", default=5, type=int, required=False, 
                        help="Number of rows in the grid of images to plot.")
    parser.add_argument("--nb_cols", default=5, type=int, required=False, 
                        help="Number of columns in the grid of images to plot.")
    parser.add_argument("--show_generated", default=True, type=bool, required=False, 
                        help="Option to show generated images")
    parser.add_argument("--save_to_dir", default="generated_by_model/generated.jpg", type=str, required=False, 
                        help="Option to save generated images to a directory")
    parser.add_argument("--noise_dim", default=100, type=int, required=False, 
                        help="Dimensionality of the noise vector to be fed to the generator. NOTE: specify noise_dim in harmony with model")
    parser.add_argument("--batch_size", default=64, type=int, required=False, 
                        help="Number of images to be generated in a single batch. NOTE: specify batch_size in harmony with model")
 
    # Parse the arguments and convert them to a dictionary
    args = vars(parser.parse_args())

    return args


def generate_animes(generator, nb_rows, nb_cols, batch_size, noise_dim, save_to_dir, show_generated):
    """
    This function generates anime faces using the generator network.

    Args:
    - generator (model): a pre-trained generator model.
    - nb_rows (int): number of rows in the grid of images to plot.
    - nb_cols (int): number of columns in the grid of images to plot.
    - batch_size (int): number of images to be generated in a single batch.
    - noise_dim (int): dimensionality of the noise vector to be fed to the generator.
    - save_to_dir (path): path to the directory where the generated images will be saved.
    - show_generated (bool): boolean to indicate whether the generated images should be shown.
    
    Returns:
    - None
    """
    # Generate random noise as input to the generator network
    noise = tf.random.normal([batch_size, noise_dim])

    # Predict the generated images using the generator network
    generated_images = generator.predict_on_batch(noise)

    # Normalize the generated images to range [0,1] for visualization
    generated_images = (generated_images+1)/2

    # Clip the values to range [0,1]
    generated_images = np.array(tf.clip_by_value(generated_images, clip_value_min=0., clip_value_max=1.))

    # Check if the generated images have only one channel (grayscale)
    if generated_images.shape[-1] == 1:
        # Remove the redundant dimension if grayscale
        generated_images = np.squeeze(generated_images)

    # Set the figure size
    plt.figure(figsize=(7,5))

    # Loop over number of images to be generated
    for i in range(nb_rows*nb_cols):
        # Plot each generated image in a subplot
        plt.subplot(nb_rows, nb_cols, i+1)
        plt.imshow(generated_images[i])
        plt.axis("off")

    # Save the generated images to a directory
    plt.savefig(save_to_dir)

    # Print the path of the saved image
    print(f"Generated images saved to {save_to_dir}")

    # Display the generated images if 'show_generated' = True
    if show_generated:
        plt.show()

# This code block is the main function of the script
if __name__ == "__main__":

    # Parse the command line arguments for generating
    generate_args = set_generate_arguments()

    # Load the generator model
    generator = tf.keras.models.load_model(generate_args["pretrained_model_path"])

    # Call the function to generate anime faces
    generate_animes(generator, generate_args["nb_rows"], generate_args["nb_cols"], generate_args["batch_size"], 
                    generate_args["noise_dim"], generate_args["save_to_dir"], 
                    generate_args["show_generated"])
