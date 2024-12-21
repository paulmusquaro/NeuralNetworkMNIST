# NeuralNetworkMNIST

## General Overview
This project demonstrates the creation of a fully connected neural network using low-level TensorFlow mechanisms. The network is trained on the MNIST dataset, a collection of handwritten digit images, to classify digits from 0 to 9. The neural network architecture consists of an input layer, two hidden layers, and an output layer.  

Key features of the neural network:
- **Activation functions**: Sigmoid for the hidden layers and Softmax for the output layer.  
- **Training method**: Stochastic Gradient Descent (SGD).  
- **Evaluation metrics**: Cross-entropy loss and accuracy.  

The final accuracy achieved is 90.2% on the test dataset.

---

## Libraries Used

`TensorFlow` is an open-source machine learning framework developed by the Google Brain Team and initially released in 2015. It is designed to handle a wide variety of machine learning tasks, from neural networks to natural language processing and computer vision. TensorFlow provides both low-level APIs for detailed customization of models and high-level APIs for quick implementation of standard tasks.  

The primary goal of TensorFlow is to facilitate numerical computations and large-scale machine learning. It uses data flow graphs, where nodes represent mathematical operations, and edges represent the data arrays (tensors) that flow between them. TensorFlow was created to help researchers and developers implement machine learning models that are efficient, scalable, and portable across different environments.

`Keras` is an open-source neural network library written in Python, initially developed by François Chollet in 2015. Keras was later integrated into TensorFlow, making it the preferred high-level API for building machine learning models. Its primary purpose is to simplify the creation of deep learning models by providing an intuitive and user-friendly interface.  

Keras is particularly well-suited for quick prototyping and experimentation. It abstracts the complexity of deep learning by offering pre-built layers, optimizers, and loss functions. While TensorFlow excels at handling low-level operations, Keras bridges the gap by enabling developers to build and train deep learning models with minimal code.  


---

## Detailed Overview
### Neural Network Architecture:
- **Input layer**: Accepts 784 features (flattened 28x28 grayscale images).  
- **Hidden layer 1**: 128 neurons with Sigmoid activation.  
- **Hidden layer 2**: 256 neurons with Sigmoid activation.  
- **Output layer**: 10 neurons with Softmax activation (for class probabilities).  

### MNIST Dataset:
- **Training data**: 60,000 images.
- **Testing data**: 10,000 images.
- **Normalization**: Pixel values are scaled to the range [0, 1].
- **Flattening**: Each 28x28 image matrix is reshaped into a 1D vector of 784 elements.  

### Training Process:
1. **Loss function**: Cross-entropy to measure the difference between predicted and actual labels.
2. **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.001.
3. **Batch size**: 256 samples per batch.
4. **Epochs**: 3000 iterations with weight updates after each batch.  

### Evaluation:
- **Accuracy on training data**: ~92.1%.  
- **Accuracy on test data**: ~90.2%.  

---

## Conda (Setup and Environment)

To make the project reproducible and ensure smooth package management, this project uses Conda as a package and environment manager. Below are the steps to set up the environment:


1. **Install Conda**:
If you haven't installed Conda yet, you can download it from the official [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) websites. Anaconda is a larger distribution with more pre-installed packages, while Miniconda is a smaller, minimal version. Choose whichever suits your needs.

2. **Create a new environment:** Open your terminal and run the following command to create a new Conda environment with Python 3.7:

    ```bash
    conda create --name new_conda_env python=3.7
    ```

3. **Activate the environment:** Once the environment is created, activate it by running:

    ```bash
    conda activate new_conda_env
    ```

4. **Install required packages (Jupyter, NumPy, MatPlotLib, Pandas, Scikit-Learn, Tensorflow, Keras and Seaborn)**

    ```bash
    conda install jupyter numpy matplotlib pandas scikit-learn tensorflow keras seaborn
    ```

5. **Run Jupyter Notebook**

    ```bash
    jupyter notebook
    ```

---

## Conclusion
This project successfully demonstrates how to build and train a neural network using low-level TensorFlow mechanisms. The model achieves a strong accuracy of 90.2% on the MNIST test dataset, validating its ability to classify handwritten digits.  

Future improvements could include:
- Experimenting with additional layers or neurons.
- Using different optimization algorithms (e.g., Adam).
- Implementing data augmentation to enhance the dataset.  