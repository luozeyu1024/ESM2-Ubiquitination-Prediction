# ESM2-Ubiquitination-Prediction

## Model

In this project, we have constructed four deep learning models for the ubiquitination site prediction task, which specifically include:

- [**DNN_LinerModel**](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/DNNLinerModel): A linear model based on fully connected layers (Dense Layer). This model directly learns the prediction rules of ubiquitination sites from raw input data.
- [**ResDNNModel**](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/ResDNNModel): A deep neural network model that introduces residual blocks (Residual Block). Through the residual learning mechanism, this model can effectively alleviate the problem of vanishing gradients, enhancing the model's ability to learn from deep structures.
- **[VAE_DNNLinerModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/VAE_DNNLinerModel)**: A composite model combining Variational Autoencoder (VAE) with DNN_LinerModel. It first uses VAE to perform feature reconstruction and dimensionality reduction on the input data, then feeds the processed features into the DNN_LinerModel for ubiquitination site prediction.
- **[VAE_ResDNNModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/VAE_ResDNNModel)**: Similar to VAE_DNNLinerModel, but in the feature extraction phase, it adopts a more complex ResDNNModel architecture, aiming to further improve the model's predictive performance.

Furthermore, corresponding supervised learning stage modules are provided for both VAE_DNNLinerModel and VAE_ResDNNModel:

- **[VAE_100_DNNLinerModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/VAE_100_DNNLinerModel)**: Based on VAE_DNNLinerModel, but removes the feature reconstruction and dimensionality reduction steps of the VAE.
- [**VAE_100_ResDNNModel**](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/VAE_100_ResDNNModel): Similar to VAE_100_DNNLinerModel, except that it uses the ResDNNModel architecture during the supervised learning phase.

## Inference_test_data

To evaluate the performance of the above models, we have prepared a test dataset:

- **[ESM2_3B_2560](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Inference_test_data/ESM2_3B_2560)**: This dataset contains raw input data without any preprocessing, suitable for testing all models (DNN_LinerModel, ResDNNModel, VAE_DNNLinerModel, VAE_ResDNNModel).
- [**ESM2_3B_VAE_100**](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Inference_test_data/ESM2_3B_VAE_100): This is the result obtained by applying VAE for feature reconstruction and dimensionality reduction on the ESM2_3B_2560 dataset. This dataset is specially designed to validate the performance of VAE_100_DNNLinerModel and VAE_100_ResDNNModel models, to explore the impact of feature dimensionality reduction on model prediction accuracy.