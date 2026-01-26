import pandas as pd
import numpy as np

def verify():
    # Load data
    data = pd.read_csv('MicrosoftStock.csv')
    stock_close = data.filter(["close"])
    dataset = stock_close.values

    # Simulate the training data length calculation from the notebook
    training_data_len = int(np.ceil(len(dataset) * 0.95))
    training_data = dataset[:training_data_len]

    # Sliding window logic
    X_train, y_train = [], []
    window_size = 60

    for i in range(window_size, len(training_data)):
        X_train.append(training_data[i-window_size:i, 0])
        y_train.append(training_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    print(f"Window size: {window_size}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Assertions
    assert X_train.shape[1] == window_size, f"Input window size should be {window_size}"
    assert len(X_train) == len(y_train), "Input and target lengths should match"

    # Check if for each X, the corresponding y is the very next day
    # X_train[0] is data[0:60], y_train[0] should be data[60]
    np.testing.assert_array_equal(X_train[0], training_data[0:60, 0])
    assert y_train[0] == training_data[60, 0]

    print("Verification successful: Sliding window is 60 days, predicting 1 day.")

if __name__ == "__main__":
    verify()
