import os
import argparse
from get_dataset import GetDataset
from models import GANModel
from mycallback import PlotImages
from warnings import filterwarnings
filterwarnings("ignore")


def set_train_arguments():
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
    parser.add_argument("--dataset_path", default=None, type=str, required=True, 
                        help="Path to image dataset")
    parser.add_argument("--height", default=64, type=int, required=False, 
                        help="Height of image")
    parser.add_argument("--width", default=64, type=int, required=False, 
                        help="Width of image")
    parser.add_argument("--nb_channels", default=3, type=int, required=False, choices=[1, 3], 
                        help="Number of channels of image (choose one of those: 1 or 3)")
    parser.add_argument("--epochs", default=100, type=int, required=False, 
                        help="Number of training epochs")
    parser.add_argument("--noise_dim", default=100, type=int, required=False, 
                        help="Dimension of the noise vector to generate fake images")
    parser.add_argument("--batch_size", default=64, type=int, required=False,
                        help="The batch size for dividing the dataset into chunks")
    parser.add_argument("--buffer_size", default=1000, type=int, required=False, 
                        help="Buffer size for shuffling the dataset")
    parser.add_argument("--nb_rows", default=5, type=int, required=False, 
                        help="Number of rows in the grid of images to plot.")
    parser.add_argument("--nb_cols", default=5, type=int, required=False, 
                        help="Number of columns in the grid of images to plot.")
    parser.add_argument("--save_epoch_results", default=True, type=bool, required=False, 
                        help="Option to save epoch results")
    parser.add_argument("--show_epoch_results", default=False, type=bool, required=False, 
                        help="Option to show epoch results")
    parser.add_argument("--patience_to_show", default=1, type=int, required=False,
                        help="Number of epochs to wait before showing the generated images, if 'show_epoch_results' == True.")
    parser.add_argument("--generator_optimizer", default="adam", type=str, required=False, choices=["sgd", "rmsprop", "adam"],
                        help="Generator model optimizer (choose one of those: sgd, rmsprop or adam)")
    parser.add_argument("--discriminator_optimizer", default="adam", type=str, required=False, choices=["sgd", "rmsprop", "adam"],
                        help="Discriminator model optimizer (choose one of those: sgd, rmsprop or adam)")
    parser.add_argument("--learning_rate", default=0.001, type=float, required=False,
                        help="The learning rate used during training.")
    parser.add_argument("--beta_1", default=0.9, type=float, required=False,
                        help="The first hyperparameter for the Adam optimizer")
    parser.add_argument("--beta_2", default=0.999, type=float, required=False, 
                        help="The second hyperparameter for the Adam optimizer")
    parser.add_argument("--epsilon", default=1e-7, type=float, required=False, 
                        help="A small constant added to the denominator to prevent division by zero")
    parser.add_argument("--momentum", default=0., type=float, required=False, 
                        help="Momentum term for the SGD optimizer")
    parser.add_argument("--nesterov", default=False, type=bool, required=False, 
                        help="Whether to use Nesterov momentum for the SGD optimizer")
    parser.add_argument("--rho", default=0.9, type=float, required=False,
                        help="Decay rate for the moving average of the squared gradient for the RMSprop optimizer") 
    parser.add_argument("--rmsprop_momentum", default=0., type=float, required=False, 
                        help="The momentum term for the RMSprop optimizer")
    
    # Parse the arguments and convert them to a dictionary
    args = vars(parser.parse_args())

    return args

if __name__ == "__main__":

    # Parse the command line arguments for training
    args = set_train_arguments()

    # Get the dataset for training
    get_dataset = GetDataset(args["height"], args["width"], args["nb_channels"], args["batch_size"], args["buffer_size"])
    dataset = get_dataset(args["dataset_path"])

    # Build GAN model
    gan_model = GANModel(args["height"], args["width"], args["nb_channels"], args["noise_dim"])

    # Compile the models
    gan_model.compile(args["generator_optimizer"], args["discriminator_optimizer"], 
                      args["learning_rate"], args["beta_1"], args["beta_2"], args["epsilon"], 
                      args["momentum"], args["nesterov"], args["rho"], args["rmsprop_momentum"])

    # Callback
    plot_images = PlotImages(args["noise_dim"], args["batch_size"], args["save_epoch_results"], 
                             args["show_epoch_results"], args["patience_to_show"], args["nb_rows"], args["nb_cols"])

    # Train the models and get results
    hist = gan_model.fit(dataset, epochs=args["epochs"], callbacks=plot_images, verbose=1)

    # Save the generator and discriminator models
    if os.path.exists("./pretrained_models") == False:
        os.makedirs("./pretrained_models")
    gan_model.generator.save("./pretrained_models/generator_model")
    gan_model.discriminator.save("./pretrained_models/discriminator_model")
    print()
    print("Trained Generator model has been saved to './pretrained_models/generator_model'")
    print("Trained Discriminator model has been saved to './pretrained_models/discriminator_model'")
