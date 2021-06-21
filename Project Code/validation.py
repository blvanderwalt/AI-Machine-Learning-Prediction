import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plot
import rnn


# def create_in_sequences_normalised_per_region(all_data, window):
#     out_seq = []
#     for in_data in all_data:
#         l = len(in_data)
#         temp_seq = []
#         for i in range(l - window):
#             train_seq = torch.FloatTensor(in_data[i:i + window])
#             train_label = torch.FloatTensor(in_data[i + window:i + window + 1])[0]

#             mx = max([max(i) for i in train_seq])

#             train_seq = torch.div(train_seq, mx)
#             train_label = torch.div(train_label, mx)
#             temp_seq.append((train_seq, train_label, mx))
#         out_seq.append(temp_seq)
#     return out_seq


model = rnn.LSTM()
model.load_state_dict(torch.load("../trained_model_state_ep50.pt"))
model.eval()


window = 28

validation_data = rnn.load_folder("../data/validation_data/")

validation_in_seq = rnn.create_in_sequences_normalised(validation_data, window)

differences = [[],[],[],[]]
count = 0

for seq, labels in validation_in_seq:
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

    mx = torch.max(seq)
    seq, labels = torch.div(seq, mx), torch.div(labels, mx)
    prediction = model(seq)
    prediction *= mx


    # Adjust Weights
    prediction[0] = prediction[0] / (1 - 0.3514)
    prediction[1] = prediction[1] / (1 - 0.2965)
    prediction[2] = prediction[2] / (1 - 0.3456)
    prediction[3] = prediction[3] / (1 - 0.3967)

    # Calculate Differences for each prediction across the Data set
    for i in range(len(prediction)):
        if labels[i] != 0:
            difference = (prediction[i] - labels[i]) / labels[i]
            differences[i].append(difference.item()*100)
            
for i in range(len(differences)):
    differences[i] = np.asarray(differences[i])

print("Confirmed: {:0.2f}% +/- {:0.2f}%".format(np.mean(differences[0]), np.std(differences[0])))
print("Deaths: {:0.2f}% +/- {:0.2f}%".format(np.mean(differences[1]), np.std(differences[1])))
print("Recovered: {:0.2f}% +/- {:0.2f}%".format(np.mean(differences[2]), np.std(differences[2])))
print("Active: {:0.2f}% +/- {:0.2f}%".format(np.mean(differences[3]), np.std(differences[3])))


# Weekly Prediction
# weekly_seq = create_in_sequences_normalised_per_region(validation_data, window)

# fut_pred = 7
# averages = [[],[],[],[]]
# totals_pred = [0,0,0,0]
# totals_real = [0,0,0,0]

# # Iterate through each region
# for region in weekly_seq:

#     input_data = region[-(fut_pred+1)][0].tolist()
#     og_data = region[-1]
#     og_max = region[-1][2].item()
#     mx = region[-(fut_pred+1)][2].item()

#     # Prediction for the week
#     # input_data[-7:] will give you the list of predictions
#     for i in range(fut_pred):
#         seq = torch.FloatTensor(input_data[-window:])

#         with torch.no_grad():
#             model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

#             prediction = model(seq)

#             # Adjust Weights
#             prediction[0] = prediction[0] / (1 - 0.3514)
#             prediction[1] = prediction[1] / (1 - 0.2965)
#             prediction[2] = prediction[2] / (1 - 0.3456)
#             prediction[3] = prediction[3] / (1 - 0.3967)

#             input_data.append(prediction.tolist())

#     sumReal = [0, 0, 0, 0]
#     sumPredict = [0, 0, 0, 0]
#     for i in range(fut_pred):
#         for k in range(4):
#             real = (og_data[0][-(fut_pred - i)][k].item() * og_max)
#             predict = (input_data[-(fut_pred - i)][k] * mx)
#             sumReal[k] += real
#             sumPredict[k] += predict

#     for i in range(4):
#         avgR = (sumReal[i] / fut_pred)
#         avgP = (sumPredict[i] / fut_pred)
#         totals_pred[i] += avgP
#         totals_real[i] += avgR
#         if avgR != 0:
#             averages[i].append( ( ( avgP - avgR ) / avgR) * 100)


# print("Last Item: Next day predicted Total: {:0f}".format((input_data[-7][0] * mx)))
# print("Last Item: Next day actual Total: {:0f}\n".format((og_data[0][-7][0].item() * og_max)))

# print("Last Item: Prediction Average: {:0.2f}".format(sumPredict[0] / fut_pred))
# print("Last Item: Next Week Average: {:0.2f}\n".format(sumReal[0]/fut_pred))

# print("Predicted Total Average Cases over the week: {:0.2f}".format(totals_pred[0]))
# print("Actual  Total Average Cases over the week: {:0.2f}\n".format(totals_real[0]))


# print("Confirmed: {:0.2f}% +/- {:0.2f}%".format(np.mean(averages[0]), np.std(averages[0])))
# print("Deaths: {:0.2f}% +/- {:0.2f}%".format(np.mean(averages[1]), np.std(averages[1])))
# print("Recovered: {:0.2f}% +/- {:0.2f}%".format(np.mean(averages[2]), np.std(averages[2])))
# print("Active: {:0.2f}% +/- {:0.2f}%".format(np.mean(averages[3]), np.std(averages[3])))


