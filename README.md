# Interactive Curve Fitting App

This is a Streamlit-based application for curve fitting and data visualization. It allows users to input or upload datasets, choose from a variety of curve types, and visualize fitted curves alongside real-time error metrics.

## Features
- **Data Input**: Enter data manually or upload a CSV file.
- **Curve Types Supported**:
  - Linear
  - Polynomial
  - Exponential
  - Gaussian
  - Arrhenius
  - Logarithmic
  - Power Law
- **Error Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
- **Interactive Visualization**: Scatter plots of original data with fitted curves.

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: For building the interactive web app.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation.
- **Matplotlib**: For plotting and visualization.
- **SciPy**: For curve fitting.
- **scikit-learn**: For error metric calculations.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/interactive-curve-fitting
   ```
2. Navigate to the project directory:
   ```bash
   cd interactive-curve-fitting
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Select your data input method (manual entry or CSV upload) in the sidebar.
2. Provide the X and Y values of your dataset.
3. Choose the curve type you want to fit.
4. View the fitted curve, parameters, and error metrics.
5. Visualize the original data and the fitted curve on an interactive plot.

## Example Dataset
Sample X values: `1, 2, 3, 4, 5`  
Sample Y values: `2.3, 2.5, 2.7, 2.9, 3.1`

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.

## Contributions
Contributions are welcome! If you find a bug or have an idea for improvement, feel free to open an issue or submit a pull request.
