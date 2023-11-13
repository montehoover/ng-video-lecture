import torch
from torch.nn import functional as F

def train(model, train_dataloader, test_dataloader, loss_fn=F.cross_entropy, optim=None, epochs=10, ):
    """
    Given a model and a dataset, train the model for a given number of epochs.
    Any loss function and optimizer can be used; cross-entropy and AdamW are the defaults.
    Consists of the typical 4-step training loop (forward, loss, backward, update), along
    with an optional validation loop. Loss values throughout the training are logged and
    checkpoints are saved to disk.
    Allows for running on GPU, CPU, or Apple M-series silicon.
    """
    if not optim:
        optim = torch.optim.AdamW(model.parameters())

    # Setup device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():  # Apple M-series silicon
        device = 'mps'
    else:
        device = 'cpu'

    # Setup logging
    train_losses = []
    val_losses = []
    best_val_loss = torch.inf

    # Main loop for train and eval
    for epoch in epochs:
        train_loss = 0
        val_loss = 0

        # Train loop
        model.train()
        for batch in train_dataloader:
            x, y = batch
            x, y, model = x.to(device), y.to(device), model.to(device)
            # 1. Forward pass
            output = model(x)
            # 2. Compute loss
            loss = loss_fn(output, y)
            # 3. Backward pass (get gradients) (By default pytorch saves gradients from last backward pass and adds new ones to them, but we don't want that in a normal (i.e. non-RNN) neural network.)
            optim.zero_grad()
            loss.backward()
            # 4. Update parameters (perform one step of gradient descent)
            optim.step()
            # Record stats
            train_loss += loss.item()    

        # Eval loop
        model.eval()
        for batch in test_dataloader:
            x, y = batch
            x, y, model = x.to(device), y.to(device), model.to(device)
            # 1. Forward pass
            output = model(x)
            # 2. Compute loss
            loss = loss_fn(output, y)
            # 3. Backward pass (get gradients)
            optim.zero_grad()
            loss.backward()
            # 4. Update parameters (perform one step of gradient descent)
            optim.step()
            # Record stats
            val_loss += loss.item()  
        
        # End of epoch stats
        train_loss = train_loss / train_dataloader.batch_size
        val_loss = val_loss / test_dataloader.batch_size
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}"
        )
        if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_{epoch+1}.pth")
    print("Finished Epochs")
    stats = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return model, stats