#import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rnn

mpl.rcParams.update({'font.size': 20})

def roundWithUncertainties(r, n=3):
    h = r[0]
    uh = r[1]
    precision = -int(math.floor(math.log10(uh))) + (n-1)
    uh = round(uh, precision)
    h = round(h, precision)
    return (h,uh)

def create_in_sequences_normalised_per_region(all_data, window):
    out_seq = []
    for in_data in all_data:
        l = len(in_data)
        temp_seq = []
        for i in range(l - window):
            train_seq = torch.FloatTensor(in_data[i:i + window])
            train_label = torch.FloatTensor(in_data[i + window:i + window + 1])[0]

            mx = max([max(i) for i in train_seq])

            train_seq = torch.div(train_seq, mx)
            train_label = torch.div(train_label, mx)
            temp_seq.append((train_seq, train_label, mx))
        out_seq.append(temp_seq)
    return out_seq

def makePlot(predicted, real, title, filename=None):
    plt.figure(figsize=(12,8))

    pred_confirmed, pred_deaths, pred_recovered, pred_active = predicted
    real_confirmed, real_deaths, real_recovered, real_active = real

    plt.plot(real_confirmed, 'b-', label="Confirmed")
    plt.plot(real_deaths, 'r-', label="Deaths")
    plt.plot(real_recovered, 'g-', label="Recovered")
    plt.plot(real_active, 'c-', label="Active")

    plt.plot(pred_confirmed, 'b--')
    plt.plot(pred_deaths, 'r--')
    plt.plot(pred_recovered, 'g--')
    plt.plot(pred_active, 'c--')

    plt.ylabel("Number of cases")
    plt.legend()

    if filename == None:
        plt.show()
    else:
        plt.savefig("./Figures/"+filename)

def predictBaseline(seq):
    prediction = [0,0,0,0]
    if len(seq) > 1:
        # first = seq[0][0].tolist()
        # last = seq[-1][0].tolist()
        first = seq[0]
        last = seq[-1]

        for i in range(len(first)):
            change = (last[i] - first[i])/(len(seq)-1)
            prediction[i] = last[i] + change

    return prediction

model = rnn.LSTM()
model.load_state_dict(torch.load("../trained_model_state_ep50.pt"))
model.eval()

window = 28
#test_data = rnn.load_folder("../data/test_data/")
test_data = [rnn.load_file("../data/test_data/Finland.csv")]

# Weekly Prediction
weekly_seq = create_in_sequences_normalised_per_region(test_data, window)

fut_pred = 7
averages = [[],[],[],[]]

totals_pred = [[],[],[],[]]
totals_base_pred = [[],[],[],[]]
totals_real = [[],[],[],[]]
weeks_pred = [0,0,0,0]
weeks_base_pred = [0,0,0,0]
weeks_real = [0,0,0,0]

# Iterate through each region
for region in weekly_seq:
    for c in range(len(region)-fut_pred):
        input_data = region[c][0].tolist()
        baseline_input_data = region[c][0].tolist()
        #input_data = region[-(fut_pred+1)][0].tolist()
        og_data = region[c+fut_pred]
        og_max = region[c+fut_pred][2].item()
        mx = region[c][2].item()

        # Prediction for the week
        # input_data[-7:] will give you the list of predictions
        for i in range(fut_pred):
            seq = torch.FloatTensor(input_data[-window:])

            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

                prediction = model(seq)
                rnn.rebalance(prediction)

                input_data.append(prediction.tolist())

            baseline_prediction = predictBaseline(baseline_input_data[-window:])
            baseline_input_data.append(baseline_prediction)

        sumReal = [0, 0, 0, 0]
        sumPredict = [0, 0, 0, 0]
        sumBaselinePredict = [0, 0, 0, 0]
        for i in range(fut_pred):
            for k in range(4):
                real = (og_data[0][-(fut_pred - i)][k].item() * og_max)
                predict = (input_data[-(fut_pred - i)][k] * mx)
                baseline_predict = (baseline_input_data[-(fut_pred - i)][k] * mx)


                sumReal[k] += real
                sumPredict[k] += predict
                sumBaselinePredict[k] += baseline_predict

        for i in range(4):
            avgR = (sumReal[i] / fut_pred)
            avgP = (sumPredict[i] / fut_pred)
            avgBP = (sumBaselinePredict[i] / fut_pred)
            weeks_pred[i] += avgP
            weeks_base_pred[i] += avgBP
            weeks_real[i] += avgR

            totals_real[i].append(avgR)
            totals_pred[i].append(avgP)
            totals_base_pred[i].append(avgBP)


error = [[(predict-real)/real for predict,real in zip(totals_pred[i], totals_real[i]) if real != 0] for i in range(4)]
error = map(np.asarray, error)
error = map(lambda x: (np.mean(x), np.std(x)), error)
error = map(roundWithUncertainties, error)
error = list(map(lambda x: x*100, error))

makePlot(totals_pred, totals_real, "Daily stuffs maybe")

print("Last Item: Next day predicted Total: {:0f}".format((input_data[-7][0] * mx)))
print("Last Item: Next day actual Total: {:0f}\n".format((og_data[0][-7][0].item() * og_max)))


#print("Last Item: Prediction Average: {:0.2f}".format(sumPredict[0] / fut_pred))
#print("Last Item: Next Week Average: {:0.2f}\n".format(sumReal[0]/fut_pred))

#print("Predicted Total Average Cases over the week: {:0.2f}".format(totals_pred[0]))
#print("Actual  Total Average Cases over the week: {:0.2f}\n".format(totals_real[0]))


print("Confirmed: {}% +/- {}%".format(*error[0]))
print("Deaths: {}% +/- {}%".format(*error[1]))
print("Recovered: {}% +/- {}%".format(*error[2]))
print("Active: {}% +/- {}%".format(*error[3]))



base_error = [[(base_predict-real)/real for base_predict,real in zip(totals_base_pred[i], totals_real[i]) if real != 0] for i in range(4)]
base_error = map(np.asarray, base_error)
base_error = map(lambda x: (np.mean(x), np.std(x)), base_error)
base_error = map(roundWithUncertainties, base_error)
base_error = list(map(lambda x: x*100, base_error))

makePlot(totals_base_pred, totals_real, "Daily stuffs maybe")

print("Last Item: Next day predicted Total: {:0f}".format((baseline_input_data[-7][0] * mx)))
print("Last Item: Next day actual Total: {:0f}\n".format((og_data[0][-7][0].item() * og_max)))

print("Confirmed: {}% +/- {}%".format(*base_error[0]))
print("Deaths: {}% +/- {}%".format(*base_error[1]))
print("Recovered: {}% +/- {}%".format(*base_error[2]))
print("Active: {}% +/- {}%".format(*base_error[3]))