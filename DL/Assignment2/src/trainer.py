import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup


def train(args, model, train_loader, device, wandb):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * 2) 
    global_step = 0

    epochs = args.n_epochs
    if args.model != 3:
        epochs = 3 # Because LSTM and Transformer model are not 'pre-trained' models

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for batch in tqdm(train_loader):
            batch = {k:v.to(device) for k, v in batch.items()}

            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            token_type_ids=batch["token_type_ids"],
                            start_positions=batch["start_positions"],
                            end_positions=batch["end_positions"])

            loss = outputs[0]
            # Calculate gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update model parameters
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            train_loss += loss
            wandb.log({"Train Loss": loss})

            if global_step % 1000 == 0:
                print (f"Loss:{loss.item()}")
        
        wandb.log({'Avg Train Loss per epoch': train_loss / len(train_loader)})

    return model

# dev_dataset for extracting real answer
def evaluate(model, valid_loader, device, wandb):
    all_start_logits = []
    all_end_logits = []

    model.eval()
    for batch in tqdm(valid_loader):
        batch = {k:v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            token_type_ids=batch["token_type_ids"],)
        start_logits, end_logits = outputs[0], outputs[1]
        for start_logit, end_logit in zip(start_logits, end_logits):
            all_start_logits.append(start_logit.cpu().numpy())
            all_end_logits.append(end_logit.cpu().numpy())

    return (all_start_logits, all_end_logits)
