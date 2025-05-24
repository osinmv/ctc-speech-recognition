import spaghetti_code_zen
import os
import torch
from pathlib import Path
from dataloader import SpeechToTextDataset
from model import MyMediumLSTMModel


def my_collate_fn(batch):
    # Assuming batch is a list of tuples (input, target) where:
    # input: (seq_len, feature_dim), target: (target_len)
    inputs, targets = zip(*batch)
    input_lengths = []
    target_lengths = []
    for i, t in batch:
        input_lengths.append(len(i))
        target_lengths.append(len(t))
    # Pad inputs to the max sequence length in the batch (for RNN)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    # Also pad target sequences to the max target length (for CTC)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return inputs_padded, targets_padded, torch.tensor(input_lengths), torch.tensor(target_lengths)



def loop(model: torch.nn.Module, optimizer:torch.optim.Optimizer, dataloader: torch.utils.data.DataLoader, is_training = True):
    model = model.train() if is_training else model.eval()
    total_loss = 0
    for _, (inputs, targets, input_lengths, target_lengths) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = model(inputs)
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()
        
        logits = logits.permute(1, 0, 2)
        loss = torch.nn.functional.ctc_loss(logits, targets, input_lengths, target_lengths, blank=0)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        if loss.isnan():
            breakpoint()
    print(f"Training({is_training}) Epoch average loss: {total_loss/len(dataloader)}")

if __name__ == "__main__":
    model = MyMediumLSTMModel.to("cuda")
    if os.getenv("CHECKPOINT"):
        model.load_state_dict(torch.load(os.getenv("CHECKPOINT")))
    dataset = SpeechToTextDataset(Path(os.getenv("TRAINDATASET")))
    test_dataset = SpeechToTextDataset(Path("TESTDATASET"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = int(os.getenv("EPOCHS", 10))
    start_epoch = int(os.getenv("START", 0))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=my_collate_fn, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, collate_fn=my_collate_fn, shuffle=False)
    for epoch in range(start_epoch, start_epoch + num_epochs):
        loop(model, optimizer, dataloader, is_training=True)
        if epoch % 5 == 0 and epoch!=start_epoch:
            loop(model, optimizer, test_dataloader, is_training=False)
        torch.save(model.state_dict(), f'model-{model.__class__.__name__}-checkpoint-{epoch}.pth')

