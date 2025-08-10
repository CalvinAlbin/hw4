from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]





class Residual(nn.Module):
    def __init__(self, dim, p_drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        #Dropping random activations
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):                 # x: (B, H)
        h = nn.Functional.relu(self.fc1(x))           # (B, H)
        h = self.drop(h)                  # (B, H)
        h = self.fc2(h)                   # (B, H)
        return nn.Functional.relu(x + h)              # (B, H)




class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden: int = 64,
        p_drop: float = 0.1
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        #dimensions
        in_dim  = n_track * 2 * 2         # 10*2*2 = 40  (left+right, x/z)
        out_dim = n_waypoints * 2         # 3*2 = 6

        # (1) input stabilization
        self.in_norm = nn.LayerNorm(in_dim)

        # stem
        self.fc_in = nn.Linear(in_dim, hidden)

        #two residual blocks, each containing 2 layers
        self.block1 = Residual(hidden, p_drop=p_drop)
        self.block2 = Residual(hidden, p_drop=p_drop)

        # head
        self.fc_out = nn.Linear(hidden, out_dim)

        # input bypass (linear shortcut input -> output)
        self.bypass = nn.Linear(in_dim, out_dim)




    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        #-----------Data Transformation-------------------------------------------------
        #getting batch size
        B = track_left.size(0)
        #concatinating the left and right inputs into one single input size (B, 20, 2)
        x = torch.cat([track_left, track_right], dim=1)
        #reshaping into B vectors of size 40 (B, 40)
        x = x.reshape(B, -1)


        #-------------Pass Data Through Model------------------------------------------
        # #First Layer
        # x = self.fc1(x)
        # #Relu
        # x = nn.functional.relu(x)
        # #Output Layer
        # x = self.fc2(x)

        # #reshaping the output to be (B, 3, 2)
        # return x.view(B, self.n_waypoints, 2)
    
        x_n = self.in_norm(x)             # (B, 40)
        h   = F.relu(self.fc_in(x_n))     # (B, 64)
        h   = self.block1(h)              # (B, 64)
        h   = self.block2(h)              # (B, 64)

        y_hat = self.fc_out(h) + self.bypass(x_n)  # (B, 6)
        return y_hat.view(B, 3, 2)        # (B, 3, 2)














class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        raise NotImplementedError


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
