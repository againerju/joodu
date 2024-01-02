import torch.nn as nn
import torch.optim
import pytorch_lightning as pl


class Ereg(pl.LightningModule):
    """Error regression network on the latent space. 
    It estimates the trajectory prediction error given 
    the latent representation of the scene encoder of
    the trajectory prediction model.    
    """

    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 32,
                 output_dim: int = 1,
                 lr: float = 1e-3,
                 ) -> None:
        super(Ereg, self).__init__()
        self.save_hyperparameters()

        self.lr = lr

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.u = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim))

        self.loss = nn.MSELoss()

    def predict_uncertainty(self, feature):
        """Method to compute the uncertainty for each feature vector.

        Args:
            features: Latent features [NxF] with number of samples N and 
                feature dimension F.
        Return:
            Uncertainty [N]
        """

        e_hat = self.u(feature)

        return e_hat

    def forward(self, batch):

        valid_agents = batch["agent_index"][batch["valid"]]

        x = batch["features"][valid_agents]
        e_hat = self.u(x)

        return e_hat

    def training_step(self, batch, batch_idx):

        h = batch["h"]
        e = batch["e"].unsqueeze(1)
        e_hat = self.u(h)

        loss = self.loss(e, e_hat)

        self.log("train_uncertainty_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, batch, batch_idx):

        h = batch["h"]
        e = batch["e"].unsqueeze(1)
        e_hat = self.u(h)

        loss = self.loss(e, e_hat)

        self.log("val_uncertainty_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=1)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('E_reg')
        parser.add_argument('--e_reg_max_epochs', type=int, default=100)
        return parent_parser
