import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define Curve Functions
def linear(x, a, b):
    return a * x + b

def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def exponential(x, a, b):
    return a * np.exp(b * x)

def gaussian(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

def arrhenius(T, A, Ea):
    R = 8.314  # Gas constant in J/(mol·K)
    return A * np.exp(-Ea / (R * T))

def logarithmic(x, a, b):
    return a * np.log(x) + b

def power_law(x, a, b):
    return a * x**b

# Sidebar: Data Input
st.sidebar.header("Data Input")
data_source = st.sidebar.radio("Data Source:", ["Upload CSV", "Enter Manually"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        x_values = data.iloc[:, 0].values
        y_values = data.iloc[:, 1].values
    else:
        st.warning("Please upload a valid CSV file.")
        x_values, y_values = np.array([]), np.array([])

elif data_source == "Enter Manually":
    x_input = st.sidebar.text_area("Enter X values (comma-separated)", "1, 2, 3, 4, 5")
    y_input = st.sidebar.text_area("Enter Y values (comma-separated)", "2.3, 2.5, 2.7, 2.9, 3.1")
    try:
        x_values = np.array([float(val.strip()) for val in x_input.split(",")])
        y_values = np.array([float(val.strip()) for val in y_input.split(",")])
    except ValueError:
        st.error("Please ensure X and Y values are valid numbers.")
        x_values, y_values = np.array([]), np.array([])

# Validate data
if len(x_values) == 0 or len(y_values) == 0:
    st.error("No valid data provided.")
elif len(x_values) != len(y_values):
    st.error("X and Y values must have the same length.")
else:
    st.success("Data successfully loaded.")

# Select Curve Type
st.header("Curve Fitting")
curve_type = st.selectbox("Choose the curve type to fit:", [
    "Linear", "Polynomial", "Exponential", "Gaussian", "Arrhenius", "Logarithmic", "Power Law"
])

# Initialize variables
fitted_curve = None
fitting_params = None

# Curve Fitting
if len(x_values) > 0 and len(y_values) > 0:
    try:
        if curve_type == "Linear":
            fitting_params, _ = curve_fit(linear, x_values, y_values)
            fitted_curve = linear(x_values, *fitting_params)
            st.write(f"Fitted Parameters: a = {fitting_params[0]:.4f}, b = {fitting_params[1]:.4f}")

        elif curve_type == "Polynomial":
            degree = st.slider("Degree of Polynomial:", 1, 10, 2)
            fitting_params = np.polyfit(x_values, y_values, degree)
            fitted_curve = np.polyval(fitting_params, x_values)
            st.write(f"Fitted Coefficients: {fitting_params}")

        elif curve_type == "Exponential":
            fitting_params, _ = curve_fit(exponential, x_values, y_values, p0=[1, 0.1])
            fitted_curve = exponential(x_values, *fitting_params)
            st.write(f"Fitted Parameters: a = {fitting_params[0]:.4f}, b = {fitting_params[1]:.4f}")

        elif curve_type == "Gaussian":
            fitting_params, _ = curve_fit(gaussian, x_values, y_values, p0=[1, np.mean(x_values), np.std(x_values)])
            fitted_curve = gaussian(x_values, *fitting_params)
            st.write(f"Fitted Parameters: a = {fitting_params[0]:.4f}, b = {fitting_params[1]:.4f}, c = {fitting_params[2]:.4f}")

        elif curve_type == "Arrhenius":
            fitting_params, _ = curve_fit(arrhenius, x_values, y_values, p0=[1, 10000])
            fitted_curve = arrhenius(x_values, *fitting_params)
            st.write(f"Fitted Parameters: A = {fitting_params[0]:.4f}, Ea = {fitting_params[1]:.4f}")

        elif curve_type == "Logarithmic":
            fitting_params, _ = curve_fit(logarithmic, x_values, y_values)
            fitted_curve = logarithmic(x_values, *fitting_params)
            st.write(f"Fitted Parameters: a = {fitting_params[0]:.4f}, b = {fitting_params[1]:.4f}")

        elif curve_type == "Power Law":
            fitting_params, _ = curve_fit(power_law, x_values, y_values)
            fitted_curve = power_law(x_values, *fitting_params)
            st.write(f"Fitted Parameters: a = {fitting_params[0]:.4f}, b = {fitting_params[1]:.4f}")

        # Error Metrics
        mae = mean_absolute_error(y_values, fitted_curve)
        mse = mean_squared_error(y_values, fitted_curve)
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"Mean Squared Error: {mse:.4f}")

        # Visualization
        fig, ax = plt.subplots()
        ax.scatter(x_values, y_values, label="Original Data", color="blue")
        if fitted_curve is not None:
            ax.plot(x_values, fitted_curve, label="Fitted Curve", color="red")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Curve fitting failed: {e}")
else:
    st.warning("Please provide valid data to fit a curve.")
