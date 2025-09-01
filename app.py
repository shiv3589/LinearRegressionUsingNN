
from flask import Flask, render_template, jsonify
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def create_and_train_model():
    # Generate synthetic data
    X = np.random.rand(100)
    y = 2 * X + 1 + np.random.randn(100) * 0.1
    
    # Build model
    model = Sequential()
    model.add(Dense(1, input_shape=(1,)))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    # Train model
    history = model.fit(X, y, epochs=100, verbose=0)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Evaluate model
    loss, mae = model.evaluate(X, y, verbose=0)
    
    return X, y, y_pred, loss, mae

def create_plot(X, y, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label='Original Data', alpha=0.7)
    plt.scatter(X, y_pred, label='Neural Network Predictions', alpha=0.7)
    plt.plot(X, 2 * X + 1, color='red', label='True Function')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression with Neural Network')
    
    # Save plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_string = base64.b64encode(img_buffer.read()).decode()
    plt.close()
    
    return img_string

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_regression')
def run_regression():
    X, y, y_pred, loss, mae = create_and_train_model()
    plot_url = create_plot(X, y, y_pred)
    
    return jsonify({
        'loss': float(loss),
        'mae': float(mae),
        'plot': plot_url
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
