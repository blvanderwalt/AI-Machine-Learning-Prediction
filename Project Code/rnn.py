"""
Steps of LSTM:
    1: Import Libraries
    2: Prepare Dataset - pre-processing
    3: Create LSTM Model
          hidden layer dimension is 100
          number of hidden layer is 1
    4: Instantiate Model
    5: Instantiate Loss
    6: Instantiate Optimizer
    7: Training the Model
    8: Prediction
"""

# Description: This program uses a LSTM RNN to predict the average
#              Active Covid-19 cases for the next week given the past numbers

# Hyper parameters
WINDOW_SIZE = 28
LEARNING_RATE = 0.0001
HIDDEN_LAYER_SIZE = 175
DROPOUT = 0.35
MAX_EPOCHS = 60
MIN_NUM_EPOCHS = 40
EPOCH_DETECT_TREND = 5

# --- Import libraries --- #
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import MinMaxScaler


# --- Start Pre-processing --- #
# Test data - replace with covid data
# @Team get the data into an 2D array, 4 values for each date
import time
def load_file(filename):
    with open(filename, "r") as file:
        file.readline()

        data = []

        for line in file:
            if line != "":
                line = line.strip().split(",")[1:]
                data.append(list(map(int, line)))
        return data

def load_folder(folder):
    data = []
    for file in os.listdir(folder):
        data.append(load_file(os.path.join(folder, file)))
    return data

def rebalance(prediction, adjustments=[-0.3514, -0.2965, -0.3456, -0.3967]):
    for i in range(len(prediction)):
        prediction[i] = prediction[i]/(1+adjustments[i])

def normalise(data):
    new = []
    for country in data:
        mx = max(country[0])
        for counts in country[1:]:
            mx = mx if mx > max(counts) else max(counts)

        for dataset in country:
            for i in range(len(dataset)):
                dataset[i] /= mx
        new.append(torch.FloatTensor(country))
    return new

# Function takes in data, return list of tuples: first element = list of 7 items to cases in past 7 days,
#                                                second = one item - num cases in 7+1 day (label)
# NOTE: CHANGE LIST OF ITEMS TO TUPLE OF 4 FOR THE VALUES WE HAVE
def create_in_sequences_normalised(all_data, window):
    out_seq = []
    for in_data in all_data:
        l = len(in_data)
        for i in range(l - window):
            train_seq = torch.FloatTensor(in_data[i:i + window])
            train_label = torch.FloatTensor(in_data[i + window:i + window + 1])[0]

            mx = max([max(i) for i in train_seq])
            
            train_seq = torch.div(train_seq, mx)
            train_label = torch.div(train_label, mx)
            out_seq.append((train_seq, train_label))
    return out_seq

def create_in_sequences(all_data, window):
    out_seq = []
    for in_data in all_data:
        l = len(in_data)
        for i in range(l - window):
            train_seq = torch.FloatTensor(in_data[i:i + window])
            train_label = torch.FloatTensor(in_data[i + window:i + window + 1])[0]

            out_seq.append((train_seq, train_label))
    return out_seq

# --- LSTM Model --- #
# NOTE: CHANGE INPUT & OUTPUT SIZE TO 4 ( 1 for each value we have)
class LSTM(nn.Module):
    def __init__ (self, input_size=4, hidden_layer_size=HIDDEN_LAYER_SIZE, outputsize=4):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        #  Used to create the LSTM and linear layers.
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Sequential(nn.Linear(hidden_layer_size, outputsize), nn.Dropout(DROPOUT))

        # Contains the previous hidden and cell state
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size), torch.zeros(1, 1, self.hidden_layer_size))


    def forward(self, in_seq):
        # Pass sequence through LSTM layer - outputs hidden, cell states & output at time step
        lstm_out, self.hidden_cell = self.lstm(in_seq.view(len(in_seq), 1, -1), self.hidden_cell)

        # Predicted number of cases stored as last item
        predictions = self.linear(lstm_out.view(len(in_seq), -1))
        return predictions[-1]

# --- End LSTM Model --- #

def main():
    # Load and normalise training data, in torch float tensors
    train_data = load_folder("../data/training_data/")
    train_data_normalized = normalise(train_data)

    # Sequence length = 7 for week structure
    train_window = WINDOW_SIZE

    train_in_seq = create_in_sequences_normalised(train_data, train_window)
    print("Data handled")
    #print(train_in_seq[:5])
    # --- End Pre-processing --- #

    # --- Loss: MES Loss & Adam Optimizer --- #
    # Instantiate Model
    model = LSTM()

    loss_func = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("NN setup")

    # --- End Loss--- #


    # --- Start Training --- #
    # @Team: Bring in the different data
    epochs = MAX_EPOCHS

    #Early Stopping
    n_epochs_stop = EPOCH_DETECT_TREND
    epochs_not_improved = 0
    min_val_loss = 0.1
    min_epoch = MIN_NUM_EPOCHS

    for i in range(epochs):
        for seq, labels in train_in_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_func(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        val_loss = single_loss.item()
        if i > min_epoch:
            if val_loss < min_val_loss:
                epochs_not_improved = 0
                min_val_loss = val_loss
                torch.save(model.state_dict(), "../trained_model_state_temp.pt")
            else:
                epochs_not_improved += 1
            if epochs_not_improved >= n_epochs_stop:
                print('Early stopping!')
                break

        # print out loss every 25 epochs
        #if i % 25 == 1:
        print(f'Epoc: {i:3} Loss: {single_loss.item():10.8f}')

    print(f'Epoc: {i:3} Loss: {single_loss.item():10.10f}')

    torch.save(model.state_dict(), "../trained_model_state.pt")
    # --- End Training --- #


    # --- Prediction --- #

    # future_pred = 7

    # Can input new country for testing here
    # test_inputs = train_data_normalized[-train_window:].tolist()

    # Make the actual prediction
    # Using the data given, predict the next day, then keep predicting this way for the next week
    model.eval()
    # for i in range(future_pred):
    #     seq = torch.FloatTensor(test_inputs[-train_window:])
    #     with torch.no_grad():
    #         model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
    #     test_inputs.append(model(seq).item())
    differences = [[],[],[],[]]
    count = 0
    #train_in_seq = create_in_sequences(train_data_normalized, train_window)
    train_in_seq = create_in_sequences_normalised(train_data, train_window)

    for seq, labels in train_in_seq:
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
        # print(model(seq))
        mx = torch.max(seq)
        seq, labels = torch.div(seq, mx), torch.div(labels, mx)
        prediction = model(seq)
        prediction *= mx
        for i in range(len(prediction)):
            if labels[i] != 0:
                difference = (prediction[i] - labels[i]) / labels[i]
                differences[i].append(difference.item()*100)
    for i in range(len(differences)):
        differences[i] = np.asarray(differences[i])

    # print(np.round(differences[0]))
    # print(np.round(differences[1]))
    # print(np.round(differences[2]))
    # print(np.round(differences[3]))

    print("Confirmed: {:0.2f}% +/- {:0.2f}%".format(np.mean(differences[0]), np.std(differences[0])))
    print("Deaths: {:0.2f}% +/- {:0.2f}%".format(np.mean(differences[1]), np.std(differences[1])))
    print("Recovered: {:0.2f}% +/- {:0.2f}%".format(np.mean(differences[2]), np.std(differences[2])))
    print("Active: {:0.2f}% +/- {:0.2f}%".format(np.mean(differences[3]), np.std(differences[3])))


    # Undo the normalization we did earlier
    # actual_preds = scaler.inverse_transform(np.array(test_inputs[train_window: ] ).reshape(-1, 1))
    # print(actual_preds)

    # --- End Prediction --- #


    # --- Plot Prediction --- #
    # x = np.arange(137, 144, 1)

    # plot.title('Month vs Passenger')
    # plot.ylabel('Total Passengers')
    # plot.grid(True)
    # plot.autoscale(axis='x', tight=True)
    # plot.plot(example_data['passengers'])
    # plot.plot(x, actual_preds)
    # plot.show()

if __name__ == "__main__":
    main()