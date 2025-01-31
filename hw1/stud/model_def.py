import torch
from tqdm import tqdm
from seqeval.metrics import f1_score, recall_score, precision_score
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy


class BiLSTMClassifier(torch.nn.Module):
    """
    Main model class, relatively flexible in order to facilitate hyperparameter tuning. More details about the model
    can be found looking at the initialization method. Training operations are performed by the `fit` method.
    """

    def __init__(
        self,
        embedding_matrix: torch.tensor = None,
        rnn_type: str = "LSTM",
        hidden_units: int = "500",
        layer_rnn: int = 1,
        layer_dense: int = 1,
        dropout_p: float = 0,
    ):
        """
        Initialization method of the model class
        :param embedding_matrix: Tensor holding the embeddings for tokens/entities. If None, a placeholder n_tokens*300
         is added, expecting embedding weights of that shape to be loaded along with the other model weights.
        :param rnn_type: The type of RNN used. 'RNN' for Vanilla RNN, 'LSTM' or 'GRU'.
        :param hidden_units: The amount of hidden units for the RNN hidden state.
        :param layer_rnn: The number of recurrent layers used.
        :param layer_dense: The number of feedforward layer used. Must be > 1, since 1 is needed to compute the logits
         from the upmost hidden state. ReLU is used as activation function. The number of hidden units is chosen
         automatically, progressively decreasing from the number of RNN hidden units to the number of classes/events.
        :param dropout_p: The dropout probability for regularization. If p>0, dropout is applied between stacked RNN
          and after all linear layers but the last.
        """
        super().__init__()  # Call parent initialization

        emb_size = (
            embedding_matrix.shape[1] if embedding_matrix is not None else 300
        )  # Embedding size
        self.param_dict = dict(  # Dictionary of parameters to unpack as arguments of recurrent modules
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

        if (
            embedding_matrix is None
        ):  # Add placeholder when matrix not passed as argument (useful for prediction time)
            embedding_matrix = torch.zeros(4530031, 300, dtype=torch.float32)
        self.tok_embedding = torch.nn.Embedding.from_pretrained(
            embedding_matrix
        )  # Token embedding layer (frozen)
        self.pos_layer = torch.nn.Embedding(
            6, 10, padding_idx=0
        )  # POS embedding layer (trained)

        self.fc_block = torch.nn.ModuleList()  # Fully connected block

        if layer_dense > 1:
            # Set difference in hidden units between subsequent dense layers
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

    def forward(
        self,
        input_data: list[
            torch.nn.utils.rnn.PackedSequence, torch.nn.utils.rnn.PackedSequence
        ],
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Forward pass for the model.
        """
        x, y = input_data  # Receive encoded sentences and encoded POS tagging sequences
        x, len_sents = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True
        )  # Pad encoded sentences for lookup
        x = self.tok_embedding(x)  # Lookup for tokens' embeddings
        y = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)[
            0
        ]  # Pad POS tagging sequences
        y = self.pos_layer(y)  # POS tag embeddings
        x = torch.concatenate(
            [x, y], dim=-1
        )  # Concatenate token and POS tagging representations
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, enforce_sorted=False, batch_first=True, lengths=len_sents
        )  # Pack everything back for recurrent block
        x = self.rnn_block(x)[0]  # Last hidden layer output
        x, len_norep = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True
        )  # Pad output of recurrent
        for layer in self.fc_block:
            x = layer(x)

        return x, len_norep

    def fit(
        self,
        epochs: int,
        learning_r: float,
        l2_regularization: float,
        dataloaders: dict[DataLoader, DataLoader],
        torch_device: torch.device,
    ) -> tuple[dict, list[float, ...], list[float, ...], list[float, ...]]:
        """
        Training and validation routine.
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

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     "max",
        #     patience=5,
        #     threshold=0.01,
        #     threshold_mode="abs",
        #     cooldown=1,
        #     min_lr=1e-4,
        #     verbose=True,
        # )  # Scheduler to check lr w.r.t. F1 score

        for _ in (p_bar := tqdm(range(epochs), total=epochs, position=0, leave=True)):
            for stage in ["train", "valid"]:
                if stage == "train":
                    batch_acc = 0  # batch counter
                    self.train()  # Set train mode for model (activating dropout if present)
                    for x_batch, y_batch, _, pos_tags, _ in dataloaders[
                        stage
                    ]:  # get observation for specific stage (train here)

                        x_batch, y_batch, pos_tags = (
                            x_batch.to(torch_device),
                            y_batch.to(torch_device),
                            pos_tags.to(torch_device),
                        )  # Move input tensors to device

                        y_pred, _ = self(
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
                        recall_accum = (
                            0  # Initialize accumulator of accuracy on validation
                        )
                        precision_accum = (
                            0  # Initialize accumulator of precision on validation
                        )
                        self.eval()  # Evaluation mode (deactivating dropout if present)
                        for (
                            x_batch,  # input embeddings
                            y_batch,  # ground truth grouped
                            rep_masks,  # rep masks to use for disentangling grouped predictions
                            pos_tags,  # input pos tag integers
                            c_y_batch,  # original ground truth to ensure fair F1 (complete ground truth, that's the c)
                        ) in dataloaders[
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
                            ]  # Ground truth for training
                            # Add to accumulator loss for single batch from validation
                            loss_accum += torch.nn.functional.cross_entropy(
                                torch.swapaxes(y_pred, 1, 2),
                                y_batch,
                                ignore_index=12,
                            ).item()

                            y_pred = torch.argmax(y_pred, -1)  # Get actual prediction
                            y_pred = (
                                y_pred.tolist()
                            )  # Turn padded predictions into a list
                            # Remove padding for predictions and repeat the elements
                            # according to the rep_mask, in order to account for multi-token entities
                            y_pred = [
                                np.repeat(sent[:sent_len], reps).tolist()
                                for sent, sent_len, reps in zip(
                                    y_pred, len_seq, rep_masks
                                )
                            ]
                            # Decode predictions and ground truth with label encoder
                            y_pred = [
                                dataloaders[stage]
                                .dataset.target_encoder.inverse_transform(labs)
                                .tolist()
                                for labs in y_pred
                            ]

                            c_y_batch = [
                                dataloaders[stage]
                                .dataset.target_encoder.inverse_transform(labs)
                                .tolist()
                                for labs in c_y_batch
                            ]

                            # Finally, compute f1_score with seqeval
                            validation_f1 += f1_score(
                                c_y_batch,
                                y_pred,
                                mode="strict",
                                average="macro",
                                scheme=IOB2,
                            )

                            recall_accum += recall_score(
                                c_y_batch,
                                y_pred,
                                mode="strict",
                                average="macro",
                                scheme=IOB2,
                            )  # Compute recall

                            precision_accum += precision_score(
                                c_y_batch,
                                y_pred,
                                mode="strict",
                                average="macro",
                                scheme=IOB2,
                                zero_division=0,
                            )  # Compute precision

            # append mean validation loss (mean over the number of batches)
            val_loss.append(loss_accum / len(dataloaders[stage]))
            seq_F1.append(validation_f1 / len(dataloaders[stage]))

            # scheduler.step(seq_F1[-1])  # LR Scheduler step with current F1 score

            if seq_F1[-1] > max(
                seq_F1[:-1], default=0
            ):  # If last F1 score better than any previous one
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
            p_bar.set_description(
                f"MOV TRAIN: {round(sum(loss_history[-len(dataloaders['train']):]) / len(dataloaders['train']), 2)}; "
                f"VAL: {round(val_loss[-1], 2)}; REC_VAL: {round(recall_accum / len(dataloaders[stage]), 2)}; "
                f"PREC_VAL:{round(precision_accum / len(dataloaders[stage]), 2)}; F1_VAL: {round(seq_F1[-1], 3)}"
            )  # Update tqdm bar description with end-of-epoch values

        return best_model, loss_history, val_loss, seq_F1
