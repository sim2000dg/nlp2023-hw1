import torch
from tqdm import tqdm
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy


class BiLSTMClassifier(torch.nn.Module):
    """
    Main model class, relatively flexible in order to facilitate hyperparameter tuning. More details about the model
    can be found looking at the initialization method. Training operations are performed by the `fit` function.
    """

    def __init__(
        self,
        embedding_matrix: torch.tensor,
        rnn_type: str,
        hidden_units: int,
        layer_rnn: int,
        layer_dense: int,
        dropout_p: float,
    ):
        """
        Initialization method of the model class
        :param embedding_matrix: Tensor holding the embeddings for tokens/entities.
        :param rnn_type: The type of RNN used. 'RNN' for Vanilla RNN, 'LSTM' or 'GRU'.
        :param hidden_units: The amount of hidden units for the RNN hidden state.
        :param layer_rnn: The number of recurrent layers used.
        :param layer_dense: The number of feedforward layer used. Must be > 1, since 1 is needed to compute the logits
         from the upmost hidden state. ReLU is used as activation function. The number of hidden units is chosen
         automatically, progressively decreasing from the number of RNN hidden units to the number of classes/events.
        :param dropout_p: The dropout probability for regularization. If p>0, dropout is applied between RNN and after
         all linear layers but the last.
        """
        super().__init__()

        emb_size = embedding_matrix.shape[1]
        self.param_dict = dict(
            zip(
                ["input_size", "hidden_size", "num_layers", "dropout", "bidirectional"],
                [emb_size + 10, hidden_units, layer_rnn, dropout_p, True],
            )
        )

        if rnn_type == "GRU":
            self.rnn_block = torch.nn.GRU(**self.param_dict)
        elif rnn_type == "LSTM":
            self.rnn_block = torch.nn.LSTM(**self.param_dict)
        elif rnn_type == "RNN":
            self.rnn_block = torch.nn.RNN(**self.param_dict)
        else:
            raise ValueError("Choose a valid recurrent neural network")

        self.tok_embedding = torch.nn.Embedding.from_pretrained(embedding_matrix)
        self.pos_layer = torch.nn.Embedding(6, 10, padding_idx=0)

        self.fc_block = torch.nn.ModuleList()

        if layer_dense > 1:
            dense_shift = (2 * hidden_units - 11) // layer_dense
            for index in range(layer_dense - 1):
                dense_in = 2 * hidden_units - dense_shift * index
                dense_out = 2 * hidden_units - dense_shift * (index + 1)
                self.fc_block.append(torch.nn.Linear(dense_in, dense_out))
                self.fc_block.append(torch.nn.Dropout(p=dropout_p))
                self.fc_block.append(torch.nn.ReLU())
        else:
            dense_out = 2 * hidden_units

        self.fc_block.append(torch.nn.Linear(dense_out, 11))

    def forward(self, input_data):
        """
        Forward pass for the model.
        """
        x, y = input_data
        x, len_sents = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.tok_embedding(x)
        y = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)[0]
        y = self.pos_layer(y)
        x = torch.concatenate([x, y], dim=-1)
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, enforce_sorted=False, batch_first=True, lengths=len_sents
        )
        x = self.rnn_block(x)[0]
        x, len_seq = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        for layer in self.fc_block:
            x = layer(x)

        return x, len_seq

    def fit(
        self,
        epochs: int,
        learning_r: float,
        l2_regularization: float,
        dataloaders: dict[DataLoader, DataLoader],
        torch_device: torch.device,
    ) -> tuple[dict, list[float, ...], list[float, ...], list[float, ...]]:
        """
        Training routine.
        :param epochs: number of epochs.
        :param learning_r: learning rate for Adam optimizer.
        :param l2_regularization: L2 regularization factor (or weight decay).
        :param dataloaders: A dictionary with training and validation dataloader.
        :param torch_device: The specific device chosen for training/validation.
        :return: Best model weights + training loss list + validation loss list + F1 score list
        """
        self.to(torch_device)  # Move model weights to device
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_r, weight_decay=l2_regularization
        )  # Adam optimizer initialization

        loss_history = list()  # Init loss history
        val_loss = list()  # List of validation losses
        seq_F1 = list()  # Init sequence of F1 validation scores

        for _ in (p_bar := tqdm(range(epochs), total=epochs, position=0, leave=True)):
            for stage in ["train", "valid"]:
                if stage == "train":
                    batch_acc = 0  # batch counter
                    self.train()  # Set train mode for model (activating dropout if present)
                    for x_batch, y_batch, _, pos_tags in dataloaders[
                        stage
                    ]:  # get observation for specific stage (train here)

                        x_batch, y_batch, pos_tags = (
                            x_batch.to(torch_device),
                            y_batch.to(torch_device),
                            pos_tags.to(torch_device),
                        )  # Move input tensors to device

                        y_pred, len_seq = self(
                            [x_batch, pos_tags]
                        )  # get predictions from model
                        y_batch = torch.nn.utils.rnn.pad_packed_sequence(
                            y_batch, batch_first=True, padding_value=12
                        )[
                            0
                        ]  # Pad ground truth
                        loss = torch.nn.functional.cross_entropy(
                            torch.swapaxes(y_pred, 1, 2),
                            y_batch,
                            ignore_index=12,
                        )  # compute categorical cross-entropy

                        loss.backward()  # Call backward propagation on the loss
                        optimizer.step()  # Move in the parameter space
                        optimizer.zero_grad()  # set to zero gradients for next iteration
                        loss_history.append(
                            loss.item()
                        )  # append current batch loss to loss_history
                        p_bar.set_postfix(
                            {"epoch batch": batch_acc, "batch loss": loss.item()}
                        )  # Update tqdm bar
                        batch_acc += 1  # Update batch counter

                else:  # epoch ended moving to validation
                    with torch.no_grad():  # we do not need gradients when calculating validation loss and accuracy
                        loss_accum = 0  # initialize accumulator of validation loss
                        validation_f1 = 0  # initialize accumulator of validation F1
                        self.eval()  # Evaluation mode (deactivating dropout if present)
                        for x_batch, y_batch, rep_masks, pos_tags in dataloaders[
                            stage
                        ]:  # Access the validation dataloader
                            # Move the input tensors to right device
                            x_batch, y_batch, pos_tags = (
                                x_batch.to(torch_device),
                                y_batch.to(torch_device),
                                pos_tags.to(torch_device),
                            )

                            y_pred, len_seq = self(
                                [x_batch, pos_tags]
                            )  # Get prediction from validation
                            y_batch = torch.nn.utils.rnn.pad_packed_sequence(
                                y_batch, batch_first=True, padding_value=12
                            )[
                                0
                            ]  # Ground truth
                            # add to accumulator loss for single batch from validation
                            loss_accum += torch.nn.functional.cross_entropy(
                                torch.swapaxes(y_pred, 1, 2),
                                y_batch,
                                ignore_index=12,
                            ).item()

                            y_pred = torch.argmax(y_pred, -1)  # Get actual prediction
                            y_batch = (
                                y_batch.tolist()
                            )  # Turn padded ground truth into a list
                            y_pred = (
                                y_pred.tolist()
                            )  # Turn padded predictions into a list
                            # Remove padding predictions and repeat the elements in y_batch and y_pred
                            # according to the rep_mask, in order to account for multi-token entities
                            y_batch = [
                                np.repeat(sent[:sent_len], reps).tolist()
                                for sent, sent_len, reps in zip(
                                    y_batch, len_seq, rep_masks
                                )
                            ]
                            y_pred = [
                                np.repeat(sent[:sent_len], reps).tolist()
                                for sent, sent_len, reps in zip(
                                    y_pred, len_seq, rep_masks
                                )
                            ]
                            # Decode y_batch and y_pred predictions
                            y_batch = [
                                dataloaders[stage]
                                .dataset.target_encoder.inverse_transform(
                                    np.array(sent)
                                )
                                .tolist()
                                for sent in y_batch
                            ]
                            y_pred = [
                                dataloaders[stage]
                                .dataset.target_encoder.inverse_transform(
                                    np.array(sent)
                                )
                                .tolist()
                                for sent in y_pred
                            ]
                            # Finally, compute f1_score with seqeval
                            validation_f1 += f1_score(
                                y_batch, y_pred, mode="strict", scheme=IOB2
                            )

                        # append mean validation loss (mean over the number of batches)
                        val_loss.append(loss_accum / len(dataloaders[stage]))
                        seq_F1.append(validation_f1 / len(dataloaders[stage]))

            if val_loss:
                if val_loss[-1] > max(val_loss):
                    best_model = self.state_dict()  # reference to model weights
                    if torch_device == torch.device("cpu"):
                        best_model = deepcopy(best_model)
                    else:
                        best_model = dict(  # Building a dict with keys referring state tensors on CPU memory
                            zip(
                                best_model.keys(),
                                [
                                    tensor.to(
                                        torch.device("cpu")
                                    )  # copy of state tensors to CPU memory
                                    for tensor in best_model.values()
                                ],
                            )
                        )
            else:
                best_model = None
            p_bar.set_description(
                f"MOV TRAIN: {sum(loss_history[-len(dataloaders['train']):]) / len(dataloaders['train'])} "
                f"VAL: {val_loss[-1]}; F1_VAL: {seq_F1[-1]}"
            )  # Update tqdm bar description with end-of-epoch values

        return best_model, loss_history, val_loss, seq_F1


if __name__ == '__main__':
    from utilities import ModelData, obs_collate, load_embeddings
    import os
    import numpy as np

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    embeddings, embedding_ind = load_embeddings(os.path.join("../../model", "embeddings.txt"))

    training_data = ModelData("../../data", embedding_ind)
    val_data = ModelData("../../data", embedding_ind, 'dev')
    train_dataloader = DataLoader(training_data, batch_size=128,
                                  shuffle=True, collate_fn=obs_collate)
    val_dataloader = DataLoader(val_data, batch_size=500,
                                collate_fn=obs_collate)
    dataloaders = {'train': train_dataloader,
                   'valid': val_dataloader}

    model = BiLSTMClassifier(embeddings, 'LSTM', 100, 2, 5, 0)
    model.fit(20, 1e-3, 1e-4, dataloaders, torch.device('cpu'))
