import torch


class BiLSTM(torch.nn.Module):
    def __init__(self, emb_size, n_out, hidden_units, layer_n, dropout_p):
        super().__init__()
        self.lstm_block = torch.nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_units,
            num_layers=layer_n,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.dense_layer = torch.nn.Linear(2 * hidden_units, n_out)

    def forward(self, x):
        x = self.lstm_block(x)[0]
        x, len_seq = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.dense_layer(x)

        return x, len_seq


if __name__ == '__main__':
    from utilities import trainer, ModelData, obs_collate, load_embeddings
    import os
    from torch.utils.data import DataLoader

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    embeddings = load_embeddings(os.path.join("../../model", "embeddings.txt"))

    training_data = ModelData("../../data", embeddings)
    val_data = ModelData("../../data", embeddings, 'dev')
    train_dataloader = DataLoader(training_data, batch_size=64,
                                  shuffle=True, collate_fn=obs_collate)
    val_dataloader = DataLoader(val_data, batch_size=200,
                                collate_fn=obs_collate)
    dataloaders = {'train': train_dataloader,
                   'valid': val_dataloader}

    trainer(BiLSTM(100, len(training_data.target_encoder.classes_),
                   500, 3, 0), 20, 1e-2, dataloaders, torch_device=torch.device('cpu'))
