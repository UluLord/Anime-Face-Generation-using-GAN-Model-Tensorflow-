import tensorflow as tf

class GANModel(tf.keras.Model):
    """
    This class builds the generator and discriminator models, and trains them.
    """
    def __init__(self, height, width, nb_channels, noise_dim, **kwargs):
        """
        Initialize the class.

        Args:
        - heigth (int): height of image.
        - width (int): width of image.
        - nb_channels (int): the number of channels in the dataset that will be trained.
        - noise_dim (int): dimension of the noise vector to generate fake images.
        """
        super(GANModel,self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.nb_channels = nb_channels
        self.noise_dim = noise_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def build_generator(self):
        """
        This function builds the generator model with the following specifications:
            * Conv2DTranspose layer.
            * Activation function as "relu" in hidden layers and "tanh" in last layer.
            * BatchNormalization after activation function.
                
        Returns:
        - generator (Tensorflow Model object): builded generator model.
        """
        # Build the generator model
        generator = tf.keras.models.Sequential([
        
            tf.keras.layers.Dense(self.height//16*self.width//16*256, input_shape=[self.noise_dim]),
            tf.keras.layers.Reshape(target_shape=[self.height//16, self.width//16, 256]),
            tf.keras.layers.BatchNormalization(),    

            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.Activation("relu"),    
            tf.keras.layers.BatchNormalization(),    

            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.Activation("relu"),    
            tf.keras.layers.BatchNormalization(),    

            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.Activation("relu"),    
            tf.keras.layers.BatchNormalization(),  

            tf.keras.layers.Conv2DTranspose(filters=self.nb_channels, kernel_size=(3,3), strides=(2,2), padding="same", activation="tanh")
        ])
        
        return generator
    
    def build_discriminator(self):
        """
        This function builds the discriminator model with the following specifications:
            * Conv2D layer.
            * Activation function as "LeakyReLU" in hidden layers and "sigmoid" in output layer.
            * Dropout except for the last two layers.
            * Last two layers are Flatten and Dense.
        
        Returns:
        - discriminator (Tensorflow Model object): builded discriminator model.
        """
        # Build the discriminator model
        discriminator = tf.keras.models.Sequential([
        
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", kernel_initializer="he_normal", input_shape=(self.height, self.width, self.nb_channels)),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", kernel_initializer="he_normal"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    
        return discriminator     

    def get_optimizer(self, model_optimizer, learning_rate, beta_1, beta_2, epsilon, momentum, nesterov, rho, rmsprop_momentum):
        """
        This function sets optimizer for a model.

        Args:
        - model_optimizer (str): model optimizer name.
        - learning_rate (float): the learning rate for the optimizer.
        - beta_1 (float): the first hyperparameter for the Adam optimizer.
        - beta_2 (float): the second hyperparameter for the Adam optimizer.
        - epsilon (float): a small constant added to the denominator to prevent division by zero.
        - momentum (float): the momentum term for the optimizer.
        - nesterov (bool): whether to use Nesterov momentum.
        - rho (float): the decay rate for the moving average of the squared gradient.
        - rmsprop_momentum (float): the momentum term for the RMSprop optimizer.
        
        Returns:
        - optimizer: model optimizer with parameters.
        """
        # Initialize optimizer variable
        optimizer = None

        # Check if the model optimizer is SGD
        if model_optimizer == "sgd":
            # Use the SGD optimizer with specified learning rate, momentum, and nesterov attributes
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                                momentum=momentum,
                                                nesterov=nesterov)
        # Check if the model optimizer is RMSprop
        elif model_optimizer == "rmsprop":
            # Use the RMSprop optimizer with specified learning rate, rho, and momentum attributes
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                                    rho=rho,
                                                    momentum=rmsprop_momentum)
        # Check if the model optimizer is Adam
        elif model_optimizer == "adam":
            # Use the Adam optimizer with specified learning rate, beta_1, beta_2, and epsilon attributes
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                 beta_1=beta_1,
                                                 beta_2=beta_2,
                                                 epsilon=epsilon)
        return optimizer

    def compile(self, generator_optimizer, discriminator_optimizer, learning_rate, beta_1, beta_2, epsilon, momentum, nesterov, rho, rmsprop_momentum):
        """
        This function compiles models before training.
        
        Args: 
        - generator_optimizer (str): the optimizer used by the generator model to perform gradient descent.
        - discriminator_optimizer (str): the optimizer used by the discriminator model to perform gradient descent.
        - learning_rate (float): the learning rate for the optimizer.
        - beta_1 (float): the first hyperparameter for the Adam optimizer.
        - beta_2 (float): the second hyperparameter for the Adam optimizer.
        - epsilon (float): a small constant added to the denominator to prevent division by zero.
        - momentum (float): the momentum term for the optimizer.
        - nesterov (bool): whether to use Nesterov momentum.
        - rho (float): the decay rate for the moving average of the squared gradient.
        - rmsprop_momentum (float): the momentum term for the RMSprop optimizer.
        """
        super(GANModel, self).compile()

        self.generator_optimizer = self.get_optimizer(generator_optimizer, learning_rate, beta_1, beta_2, epsilon, momentum, nesterov, rho, rmsprop_momentum)

        self.discriminator_optimizer = self.get_optimizer(discriminator_optimizer, learning_rate, beta_1, beta_2, epsilon, momentum, nesterov, rho, rmsprop_momentum)

    def generator_loss(self, fake_output):
        """
        This function computes the generator model loss.
        
        Args: 
        - fake_output (tensor): predictions on fake images.
        
        Returns:
        - gen_loss (tensor): loss of the generator model.
        """
        # Labels for fake images like real images
        fake_labels = tf.ones_like(fake_output)
        # Compute loss
        gen_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_output)
        return gen_loss

    def discriminator_loss(self, real_output, fake_output):
        """
        This function computes the discriminator model loss.
        
        Args: 
        - real_output (tensor): predictions on real images.
        - fake_output (tensor): predictions on fake images.
        
        Returns:
        - total_loss (tensor): total loss of the discriminator model.
        """
        # Labels for real images
        real_labels = tf.ones_like(real_output)
        # Labels for fake images
        fake_labels = tf.zeros_like(fake_output)
        # Compute loss for real labels and output
        real_loss = tf.keras.losses.binary_crossentropy(real_labels, real_output) 
        # Compute loss for fake labels and output
        fake_loss = tf.keras.losses.binary_crossentropy(fake_labels, fake_output)
        # All loss
        total_loss = real_loss + fake_loss

        return total_loss

    @tf.function    
    def train_step(self, dataset):
        """
        This function trains the GAN model on given dataset.
    
        Args:
        - dataset (dataset): dataset to use for training.

        Returns:
        - loss (dict): stores generator and discriminator losses.
        """
        # Get batch size
        batch_size = tf.shape(dataset)[0]
        # Create a noise
        noise = tf.random.normal(shape=(batch_size,self.noise_dim))
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            # Generate fake images with the generator model
            generated_image = self.generator(noise, training=True)
        
            # Predict for real image by using discriminator model
            real_output = self.discriminator(dataset, training=True)
            
            # Predict for generated image by using discriminator model
            fake_output = self.discriminator(generated_image, training=True) 
            
            # Compute generator and discriminator losses
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        # Update gradients of generator model
        generator_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_grad, self.generator.trainable_variables))

        # Update gradients of discriminator model
        discriminator_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grad, self.discriminator.trainable_variables))

        # Store the losses
        hist = {"generator_loss": gen_loss, "discriminator_loss": disc_loss}
        return hist