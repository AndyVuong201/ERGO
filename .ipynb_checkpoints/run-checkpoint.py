import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
import torch.optim as optim
from random import shuffle
import time
import numpy as np
import torch.autograd as autograd
from sklearn.metrics import roc_auc_score, roc_curve
import pickle
import argparse
import csv
from sklearn.model_selection import train_test_split
import os




class DoubleLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout, device):
        super(DoubleLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        # Embedding matrices - 20 amino acids + padding
        self.tcr_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        # MLP
        self.hidden_layer = nn.Linear(lstm_dim * 2, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, lstm, padded_embeds, lengths):
    
    
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        
        lengths = lengths.cpu()                              ################ added this
        
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        unperm_idx = unperm_idx.to(lengths.device)                             ################ added this
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, tcrs, tcr_lens, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        tcr_embeds = self.tcr_embedding(tcrs)
        # LSTM Acceptor
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embeds, tcr_lens)
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1] for i, j in enumerate(tcr_lens)]).view(len(tcr_lens), self.lstm_dim)

        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(peps)
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens), self.lstm_dim)

        # MLP Classifier
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output

def convert_data(tcrs, peps, amino_to_ix):
    for i in range(len(tcrs)):
        tcrs[i] = [amino_to_ix[amino] for amino in tcrs[i]]
    for i in range(len(peps)):
        peps[i] = [amino_to_ix[amino] for amino in peps[i]]


def get_batches(tcrs, peps, signs, batch_size):
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs):
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
    # Return list of all batches
    return batches


def get_full_batches(tcrs, peps, signs, batch_size, amino_to_ix):
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs) // batch_size * batch_size:
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
    # pad data in last batch
    missing = batch_size - len(tcrs) + index
    if missing < batch_size:
        padding_tcrs = ['A'] * missing
        padding_peps = ['A' * (batch_size - missing)] * missing
        convert_data(padding_tcrs, padding_peps, amino_to_ix)
        batch_tcrs = tcrs[index:] + padding_tcrs
        batch_peps = peps[index:] + padding_peps
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        batch_signs = [0.0] * batch_size
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
        # Update index
        index += batch_size
    # Return list of all batches
    return batches
    pass


def pad_batch(seqs):
    # Tensor of sequences lengths
    lengths = torch.LongTensor([len(seq) for seq in seqs])
    # The padding index is 0
    # Batch dimensions is number of sequences * maximum sequence length
    longest_seq = max(lengths)
    batch_size = len(seqs)
    # Pad the sequences. Start with zeros and then fill the true sequence
    padded_seqs = autograd.Variable(torch.zeros((batch_size, longest_seq))).long()
    for i, seq_len in enumerate(lengths):
        seq = seqs[i]
        padded_seqs[i, 0:seq_len] = torch.LongTensor(seq[:seq_len])
    # Return padded batch and the true lengths
    return padded_seqs, lengths


def train_epoch(batches, model, loss_function, optimizer, device):
    model.train()
    shuffle(batches)
    total_loss = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch

 
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        batch_signs = torch.tensor(batch_signs).to(device)
        model.zero_grad()
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
 

        batch_signs = batch_signs.unsqueeze(1).float()
        probs = probs.float()
        

        loss = loss_function(probs, batch_signs)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    return total_loss / len(batches)


def train_model(batches, test_batches, device, args, params, loss_fun):
    losses = []
    # We use Cross-Entropy loss

    
    loss_function = nn.BCELoss()

    
    if loss_fun == 'Sl1':
        print('confirmed training sl1 loss')
        loss_function = nn.SmoothL1Loss()
    elif loss_fun == 'Hub':
        print('confirmed training hub loss')
        loss_function = nn.HuberLoss()
    elif loss_fun == 'BCELog':
        print('confirmed training BCELOG loss')
        loss_function = nn.BCEWithLogitsLoss()
        

    
    
    # Set model with relevant parameters

    print("defining model")
    model = DoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device)
    # Move to GPU
    
    print("model - to device start")
    
    model.to(device)
    
    
    print("model - to device end, optimizer start")
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    print("optimizer end")
    
    # Train several epochs
    best_auc = 0
    best_roc = None
    for epoch in range(params['epochs']):
        print('epoch:', epoch + 1)
        epoch_time = time.time()
        # Train model and get loss
        loss = train_epoch(batches, model, loss_function, optimizer, device)
        losses.append(loss)
        # Compute auc
        
        if params['option'] == 2:
            test_w, test_c = test_batches
            test_auc_w = evaluate(model, test_w, device)
            print('test auc w:', test_auc_w)
           
            test_auc_c = evaluate(model, test_c, device)
            print('test auc c:', test_auc_c)
           
        else:
            test_auc, roc = evaluate(model, test_batches, device)

            # nni.report_intermediate_result(test_auc)

            if test_auc > best_auc:
                best_auc = test_auc
                best_roc = roc
          
           
        print('one epoch time:', time.time() - epoch_time)
    return model, best_auc, best_roc


def evaluate(model, batches, device):
    model.eval()
    true = []
    scores = []
    shuffle(batches)
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        # print(np.array(batch_signs).astype(int))
        # print(probs.cpu().data.numpy())
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
    # Return auc score
    auc = roc_auc_score(true, scores)
    fpr, tpr, thresholds = roc_curve(true, scores)
    return auc, (fpr, tpr, thresholds)




def lstm_get_lists_from_pairs(pairs):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label = pair
        tcrs.append(tcr)
        peps.append(pep[0])
      
        signs.append(float(label))
        
    #print(signs)
    return tcrs, peps, signs


def main(args, epochs, lstm_dim, emb_dim, datafile, loss_fun):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    arg = {}
    
  
    arg['siamese'] = False
    params = {}
    params['lr'] = 1e-4
    params['wd'] = 0
    params['epochs'] = epochs#10#100
    params['batch_size'] = 50
    params['lstm_dim'] = lstm_dim#50#500
    params['emb_dim'] = emb_dim#4#10
    params['dropout'] = 0.1
    params['option'] = 0
    params['enc_dim'] = 100
    params['train_ae'] = True
   
    

    all_pairs = []
    print("beginning file read")
    with open(datafile, 'r', encoding='unicode_escape') as file:
        reader = csv.reader(file)
        print("file opened")
        for line in reader:
            if len(line) < 3:  # Ensure the line contains TCR, peptide, and label.          
                continue
            pep, tcr, label = line        

            amino_acids = set('ARNDCEQGHILKMFPSTWYV')
            x = True
            for char in tcr: 
                if char not in amino_acids:
                    x = False
            for char in pep: 
                if char not in amino_acids:
                    x = False
            if x == True:
                all_pairs.append((tcr, pep, int(label)))


    print("file closed")
    #all_pairs = random.sample(all_pairs, 10000) # only random 100000   for testing locally
    
    # Split into training and test sets.     
    train, test = train_test_split(all_pairs)
    print("data split")

    # train
    train_tcrs, train_peps, train_signs = lstm_get_lists_from_pairs(train)
    convert_data(train_tcrs, train_peps, amino_to_ix)
    train_batches = get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])
    print("data batched")

    # test
    test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
    convert_data(test_tcrs, test_peps, amino_to_ix)
    test_batches = get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])
    # Train the model
    print("starting train")
    model, best_auc, best_roc = train_model(train_batches, test_batches, args.device, arg, params, loss_fun)
    print("training complete")
    pass

    # Save trained model   
    torch.save({
                'model_state_dict': model.state_dict(),
                'params': params
                }, args.model_file)

    pass



def test_model(test_data_file, model_file, device, ind, lstm_dim, emb_dim, split_type, loss_fun):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
     
    # Load test data
    tcrs, peps, labels = [], [], []

    
    with open(test_data_file, 'r', encoding='unicode_escape') as csv_file:    
        reader = csv.reader(csv_file)
        for line in reader:
            if len(line) < 3:  # Ensure the line contains TCR, peptide, and label
                continue
            pep, tcr, label = line[:3]
            
            amino_acids = set('ARNDCEQGHILKMFPSTWYV')
            x = True
            for char in tcr: 
                if char not in amino_acids:
                    x = False
            for char in pep: 
                if char not in amino_acids:
                    x = False
            if x == True:
                tcrs.append(tcr)
                peps.append(pep)
                labels.append(float(label))
            
    # Load model
    model = DoubleLSTMClassifier(emb_dim, lstm_dim, 0.1, device)# model = DoubleLSTMClassifier(10, 500, 0.1, device)
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Prepare data batches
    og_tcrs = tcrs.copy()
    og_peps = peps.copy()
    
    convert_data(tcrs, peps, amino_to_ix)
    test_batches = get_full_batches(tcrs, peps, labels, batch_size=50, amino_to_ix=amino_to_ix)

    # Make predictions and collect probabilities
    preds, all_labels = [], []

    test_list = []
    with torch.no_grad():
        for batch in test_batches:
            padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
            padded_tcrs = padded_tcrs.to(device)
            tcr_lens = tcr_lens.to(device)
            padded_peps = padded_peps.to(device)
            pep_lens = pep_lens.to(device)
            probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
            test_list.extend(probs.cpu().data)
            preds.extend([t[0] for t in probs.cpu().data.tolist()])
            all_labels.extend(np.array(batch_signs).astype(int))
    
    # Calculate metrics
    binary_preds = [1 if p >= 0.5 else 0 for p in preds] #preds
    accuracy = accuracy_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, preds)
    f1 = f1_score(all_labels, binary_preds)



    with open(f'predictions/{split_type}_{loss_fun}_test_prediction_{ind}.txt', mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['TCR', 'Peptide', 'Label', 'Prediction'])  # Add a header row
        for tcr, pep, label, pred in zip(og_tcrs, og_peps, all_labels, binary_preds):
            writer.writerow([tcr, pep, label, pred])



    with open(results_file, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        model_name = f"{split_type}_{loss_fun}_model_{ind}"
        writer.writerow([model_name, accuracy, recall, precision, auc, f1])
  
    pass





results_file = "test_results.csv"
with open(results_file, mode="a", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Model", "Accuracy", "Recall", "Precision", "AUC", "F1"])


epochs = 100
lstm_dim = 500
emb_dim = 10
device = torch.device('cuda:0')
torch.cuda.synchronize()

split_types = ['tcr','epi']
loss_functions = ['BCE','Sl1', 'Hub', 'BCELog']#         

for loss_fun in loss_functions:
    for split_type in split_types:
        for job_index in range(0,5): 

            print(f"{split_type}_{loss_fun}_{job_index}")
   
            train_data_file = fr'data/BAP/{split_type}_split/train.csv'
            test_data_file = fr'data/BAP/{split_type}_split/test.csv'

            print(f"begining model train for run number {job_index} for type {split_type}")
            args = argparse.Namespace(model_file=f'models/{split_type}_{loss_fun}_model_{job_index}.pt', device= device, function='train')
            main(args, epochs, lstm_dim, emb_dim, train_data_file, loss_fun)



            print(f"begining model test for run number {job_index} for type {split_type}")
            model_file = f'models/{split_type}_{loss_fun}_model_{job_index}.pt'
            test_model(test_data_file, model_file, device, job_index, lstm_dim, emb_dim,split_type, loss_fun)


print("ended script run")
