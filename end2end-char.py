import numpy as np
import torch

from kan import KAN
import pickle

print('Data Loading')

train_data = np.memmap("data/shakespeare_char/train.bin", np.uint16 , mode="r")
val_data = np.memmap("data/shakespeare_char/val.bin", np.uint16 , mode="r")
meta = pickle.load(open("data/shakespeare_char/meta.pkl", "rb"))

batch_size = 48
block_size = 128
embedding_dim = 64
hidden_dim = 64

# Never tested this option, purely experimental
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y



token_dim = len(meta['stoi'])
model_kan = KAN(width=[embedding_dim, hidden_dim, embedding_dim], grid=16, k=3, noise_scale=1e-2,bias_trainable=False, device=device, seed=0,vocab_size=token_dim,block_size=block_size)

print('Initialized KAN model')
print(model_kan)

x,y = get_batch("train")
x_val,y_val = get_batch("val")
print(x.shape, y.shape)

print('Training')
batch_val = get_batch("val")
for i in range(1):
    batch_train = get_batch("train")
    dataset = {"train_input": batch_train[0], "train_label": batch_train[1], 'test_input':batch_val[0], 'test_label':batch_val[1]}

    train_loss = model_kan.train(dataset, opt="Adam", steps=1)
    print(f"loss: {np.array(train_loss['train_loss']).mean()}")


print('Forward')
o = model_kan.forward(x_val)

print('Generate')
gen = model_kan.generate(x_val[0], 20)

out_str = ''.join([meta['itos'][i] for i in gen])
print(out_str)
print('Test Ended')

