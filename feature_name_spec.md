# Feature: Tkinter Data Loader and Preprocessing GUI

## 1. Objective
Build a Tkinter graphical interface that allows users to upload training and testing CSV files, select target labels, and execute the data preprocessing pipeline (handling non-numeric data, missing values, and normalization).

## 2. Inputs & Outputs (The Contract)
* **Inputs:** * User file paths via `tkinter.filedialog`.
  * User UI selections (dropdowns/checkboxes) for preprocessing methods.
* **Outputs:** * Renders a Tkinter window.
  * Outputs preprocessed `x_data` and `y_data` arrays into the application's memory/state.

## 3. Core Logic
* **Step 1: File Selection.** Provide buttons to open file dialogs for training and testing data.
* **Step 2: Load Data.** Call `load_data_train_test` from `data_prep.py` and extract column names.
* **Step 3: UI Generation.** Dynamically generate a dropdown menu populated with the extracted column names to let the user select the `labels` column.
* **Step 4: Preprocessing Options.** Render checkboxes/radio buttons mapping to the TODOs in `data_prep.py`:
  * "Encode non-numeric columns" (Triggers `encode_non_numeric`)
  * "Fill missing values with median" (Triggers `fill_missing_values`)
  * "Normalize Data (Z-score)" (Triggers `normalize_data_z_score`)
* **Step 5: Execution.** Provide a "Process Data" button that sequentially runs the selected functions and passes the final DataFrame to `split_reshape_data`.

## 4. Constraints & Error Handling
* **Must NOT:** Do not initialize or train the neural network in this module.
* **Errors:** * If the user clicks "Process Data" before uploading files, raise a UI warning popup.
  * If the selected label column does not exist, raise a `ValueError` and display an error popup.

## 5. Dependencies
* `tkinter` (Standard library)
* `data_prep.py` (`load_data_train_test`, `encode_non_numeric`, `fill_missing_values`, `normalize_data_z_score`, `split_reshape_data`)


# Feature: Generalized Neural Network Architecture

## 1. Objective
Refactor `init_network` and `feed_forward` in `Network_tools.py` to support an arbitrary number of layers with varying node counts. Eliminate all global variables (`hidden_layers`, `features`, `outputs`).

## 2. Inputs & Outputs (The Contract)
* **Inputs (init_network):** `layer_dims` (list of int). Example: `[10, 64, 32, 1]` represents input size 10, two hidden layers of 64 and 32, and output size 1. `alpha` (float), `beta` (float).
* **Outputs (init_network):** `weights` (list of numpy arrays), `biases` (list of numpy arrays).
* **Inputs (feed_forward):** `input_layer` (numpy array), `weights` (list of numpy arrays), `biases` (list of numpy arrays).
* **Outputs (feed_forward):** `cache` (list of numpy arrays representing activations for each layer).

## 3. Core Logic
* **Step 1:** Delete global variables at the top of the file.
* **Step 2: init_network:** Iterate through `layer_dims` from index 1 to the end. For layer $l$, initialize weight matrix $W$ with shape `(layer_dims[l], layer_dims[l-1])` and bias vector $b$ with shape `(layer_dims[l], 1)`.
* **Step 3: feed_forward:** Calculate $Z = W \cdot A + b$ and $A = \text{sigmoid}(Z)$ dynamically by iterating over the length of the `weights` list, rather than using a hardcoded `num_hidden_layers` variable.

## 4. Constraints & Error Handling
* **Must NOT:** Change the `sigmoid` activation function or introduce new dependencies.
* **Errors:** If `layer_dims` has fewer than 2 elements, raise a `ValueError` indicating the network must have at least an input and output layer.

## 5. Dependencies
* `numpy`