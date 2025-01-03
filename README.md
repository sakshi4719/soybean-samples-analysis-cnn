# CNN Model for Soybean Yield Prediction 

### Overview
This project uses a Convolutional Neural Network (CNN) to predict soybean yield based on soil data features. The goal is to predict the soybean yield using these features, providing valuable insights for agricultural decision-making.

### Data Dictionary
The dataset `corn_samples.csv` includes the following features:

- **bdod**: Bulk Density
- **cec**: Cation Exchange Capacity at pH 7
- **cfvo**: Coarse Fragments
- **clay**: Clay Content
- **nitrogen**: Nitrogen Content
- **ocd**: Organic Carbon Density
- **ocs**: Organic Carbon Stock
- **phh2o**: pH in Water
- **sand**: Sand Content
- **silt**: Silt Content
- **soc**: Soil Organic Carbon
- **yield**: Soybean Yield (Target Variable)

### Model Architecture
The model utilizes a CNN combined with LSTM layers to capture both spatial and temporal features of the data:

1. **Conv1D Layers**: Used to learn spatial patterns from the input soil features.
2. **LSTM Layer**: A Bidirectional LSTM is applied to capture sequential dependencies.
3. **Dense Layers**: Fully connected layers that refine the learned features.
4. **Dropout Layers**: Regularization technique to reduce overfitting.
5. **BatchNormalization**: Helps with faster convergence and improves model stability.

### Preprocessing
- The input data is scaled using `MinMaxScaler` to ensure that all features are within the range [0, 1].
- The target variable (yield) is also scaled using `MinMaxScaler` for better model performance.

### Training
The model is trained using the following settings:
- **Optimizer**: Adam with a learning rate of 0.0001.
- **Loss Function**: Mean Squared Error (MSE).
- **Callbacks**: 
  - **ReduceLROnPlateau**: Reduces learning rate when the validation loss plateaus.
  - **EarlyStopping**: Stops training early if the validation loss does not improve after 10 epochs.

### Performance Metrics
- **Final Test Loss**: The loss computed after the final epoch.
- **RMSE**: The root mean square error of the predicted yield versus the actual yield.
- **Correlation Coefficient**: The strength of the linear relationship between the predicted and actual yields.

### Results
- **Correlation Coefficient**: 0.9051 (Strong positive correlation between predicted and actual yields).
- **RMSE**: 5.479 (The model's prediction error).

### How to Use
1. Ensure that the `corn_samples.csv` dataset is available in the same directory as the script.
2. Install the required libraries using:
   ```bash
   pip install pandas scikit-learn tensorflow numpy
   ```
3. Run the script to train the model and get performance metrics.
