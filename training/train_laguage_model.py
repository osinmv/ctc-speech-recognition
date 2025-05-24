import torch
from model import CharLM
import os
from pathlib import Path
from dataloader import TextToTextDataset
from main import my_collate_fn


model = CharLM(28, 128, 128).to("cuda")
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters())
dataset = TextToTextDataset(Path(os.getenv("TRAINDATASET")))
test_dataset = TextToTextDataset(os.getenv("TESTDATASET"))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=my_collate_fn, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, collate_fn=my_collate_fn, shuffle=True)
for epoch in range(int(os.getenv("EPOCHS", 30))):
    current_dataloader = dataloader
    training = True
    total_loss = 0
    model.train()
    if epoch % 2 == 0 and epoch != 0:
        training = False
        current_dataloader = test_dataloader
        model.eval()
    for batch in current_dataloader:
        inputs, targets, _, _ = batch
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        logits, _ = model(inputs)
        loss = loss_fn(logits.view(-1, 28), targets.view(-1))
        if(training):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss+=loss.item()
    print(f'Training {training} Loss {total_loss/len(current_dataloader)}')
    torch.save(model.state_dict(), f'models/model-{model.__class__.__name__}-checkpoint-{epoch}.pth')