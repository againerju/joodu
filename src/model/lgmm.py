import joblib
import numpy
import sklearn.mixture


class LGMM(object):
    """ 
    Gaussian mixture model on the latent space. 
    It predicts the out-of-distribution score given 
    the latent representation of the scene encoder of
    the trajectory prediction model.  

    """

    def __init__(
        self, 
        model_path: str = None,
        ) -> None:

        self.model_path = None
        self.model = None

        if model_path:
            self.set_model_path(model_path)
            self.load_model()


    def set_model_path(self, model_path):
        self.model_path = model_path


    def load_model(self):
        self.model = joblib.load(self.model_path)


    def predict_ood_score(self, features: numpy.ndarray) -> numpy.ndarray:
        """Method to compute the negative log-likelihood of each feature vector.
        
        Args:
            features: Latent features [NxF] with number of samples N and 
                feature dimension F.
        Return:
            Out-of-distribution score [N]
        """
        alpha_hat = -self.model.score_samples(features)
        return alpha_hat
