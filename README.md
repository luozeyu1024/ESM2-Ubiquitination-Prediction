# ESM2-Ubiquitination Prediction

## Models

In this project, we have constructed four deep learning models for the ubiquitination site prediction task, specifically including:

- **[DNNLinerModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/DNNLinerModel)**: A linear model based on fully connected layers (Dense Layer). This model directly learns the prediction rules of ubiquitination sites from raw input data.
- **[ResDNNModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/ResDNNModel)**: A deep neural network model that introduces residual blocks (Residual Block). Through the residual learning mechanism, this model can effectively alleviate the problem of vanishing gradients, enhancing the model's ability to learn from deep structures.
- **[cVAE_DNNLinerModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/cVAE_DNNLinearModel)**: A conditional VAE model combining a Residual Variational Autoencoder (ResVAE) with DNN_LinerModel as the classification head. This model framework is trained with both reconstruction and classification objectives. During prediction, the features are directly input into the model framework, and ubiquitination site prediction is performed in the classification head DNN_LinerModel.
- **[cVAE_ResDNNModel](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Model/cVAE_ResDNNModel)**: A conditional VAE model combining a Residual Variational Autoencoder (ResVAE) with ResDNNModel as the classification head. This model framework is trained with both reconstruction and classification objectives. During prediction, the features are directly input into the model framework, and ubiquitination site prediction is performed in the classification head ResDNNModel.

## Inference Test Data

To evaluate the performance of the above models, we have prepared a test dataset:

- **[ESM2_3B_2560](https://github.com/EUP-laboratory/ESM2-Ubiquitination-Prediction/tree/main/Inference_test_data/ESM2_3B_2560)**: This dataset contains raw input data without any preprocessing, suitable for testing all models.