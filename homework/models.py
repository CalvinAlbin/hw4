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
        h = F.relu(self.fc1(x))           # (B, H)
        h = self.drop(h)                  # (B, H)
        h = self.fc2(h)                   # (B, H)
        return F.relu(x + h)              # (B, H)




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
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model


        # turn coords (x,z) into d-dim features
        self.coord_proj = nn.Linear(2, d_model)              # (...,2) -> (...,d)


        # embeddings for position (0..T-1) and side (0=left, 1=right)
        self.pos_emb  = nn.Embedding(n_track, d_model)       # (10, d)
        self.side_emb = nn.Embedding(2, d_model)             # (2, d)


        # learned queries (one per waypoint)
        self.query_emb = nn.Embedding(n_waypoints, d_model)  # (3, d)


        # --- layer norms (pre-activations)
        self.tokens_ln  = nn.LayerNorm(d_model)
        self.queries_ln = nn.LayerNorm(d_model)


        # single cross-attention layer (queries attend over tokens)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)


        # --- small FFN with residual
        #Mixing the features within the tokens
        self.ffn_fc1 = nn.Linear(d_model, 2 * d_model)
        self.ffn_fc2 = nn.Linear(2 * d_model, d_model)
        self.ffn_drop = nn.Dropout(dropout)


        # --- post-block norms
        self.post_attn_ln = nn.LayerNorm(d_model)
        self.post_ffn_ln  = nn.LayerNorm(d_model)


        # map per-query features to (x,z)
        self.head = nn.Linear(d_model, 2)


    def _make_tokens(self, left, right):
            #take the ground truth left right tensors, combine them, expand them to d dimensions
            #   create position id's for each token at side and position level, embed(lookup) the
            #   d size vector for each token's relative side and position, then comibne them
            #   with the coordinate vectors to get the final tokens which are used to create the
            #   keys and values which the queries use in the attention algorithm to make decisions
            #   the queries score each token based on keys, then compute weights using the softmax
            #   of the scores, then mix the values and weights which come together using the final
            #   simple layer to form a prediction for the query!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #   !!!!!!!!!!!!!!!!!!!! GOD THESE TRANSFORMERS ARE GOING TO KILL ME!!!!!!!!!!!!!!!!!!!!
            # left,right: (B,10,2)
            B, T, _ = left.shape
            device = left.device

            # 1) concat left then right
            tokens_2d = torch.cat([left, right], dim=1)          # (B, 20, 2)

            # 2) project coords
            coord_feat = self.coord_proj(tokens_2d)              # (B, 20, d)

            # 3) position ids: [0..T-1, 0..T-1]
            pos = torch.arange(T, device=device)                 # (10,)
            pos_ids = torch.cat([pos, pos], dim=0)               # (20,)
            pos_ids = pos_ids.unsqueeze(0).expand(B, -1)         # (B, 20)

            # 4) side ids: [0...0, 1...1]
            left_side  = torch.zeros(B, T, dtype=torch.long, device=device)
            right_side = torch.ones (B, T, dtype=torch.long, device=device)
            side_ids = torch.cat([left_side, right_side], dim=1) # (B, 20)

            # 5) add embeddings
            pos_vecs  = self.pos_emb(pos_ids)                    # (B, 20, d)
            side_vecs = self.side_emb(side_ids)                  # (B, 20, d)
            tokens = coord_feat + pos_vecs + side_vecs           # (B, 20, d)

            # 6) normalize
            tokens = self.tokens_ln(tokens)                      # (B, 20, d)
            return tokens



    def _make_queries(self, B, device):
        #Each query gets its own ID, which is then embedded (they lookup their corresponding
        #   set of features) which gets expanded across all batches, which then gets layer normalized.
        # 1) ids 0..W-1
        q_ids = torch.arange(self.n_waypoints, device=device)  # (3,)
        # 2) lookup
        q = self.query_emb(q_ids)                              # (3, d)
        # 3) batch expand
        q = q.unsqueeze(0).expand(B, -1, -1)                   # (B, 3, d)
        # 4) normalize
        q = self.queries_ln(q)                                 # (B, 3, d)
        return q

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
        # inputs: (B,T,2)
        B = track_left.size(0)
        device = track_left.device

        # --- make K,V and Q
        tokens = self._make_tokens(track_left, track_right)    # (B, 20, d)
        queries = self._make_queries(B, device)                # (B, 3, d)

        # --- cross-attention (residual)
        #                       queries, keys,   values
        attn_out, _ = self.attn(queries, tokens, tokens)       # (B, 3, d)
        x = queries + self.attn_drop(attn_out)                 # (B, 3, d)
        x = self.post_attn_ln(x)                               # (B, 3, d)

        # --- FFN (residual)
        h = self.ffn_fc1(x)                                    # (B, 3, 2d)
        h = F.relu(h)                                          # (B, 3, 2d)
        h = self.ffn_drop(h)                                   # (B, 3, 2d)
        h = self.ffn_fc2(h)                                    # (B, 3, d)
        x = self.post_ffn_ln(x + self.ffn_drop(h))             # (B, 3, d)

        # --- head: per-query to (x,z)
        out = self.head(x)                                     # (B, 3, 2)
        return out










class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)


        # Block 1: (B, 3, 96, 128) -> (B, 32, 48, 64) down
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: (B, 32, 48, 64) -> (B, 64, 24, 32) down
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3: (B, 64, 24, 32) -> (B, 128, 12, 16) up
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # Block 4: (B, 128, 12, 16) -> (B, 128, 6, 8) 
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        # Global average pool: (B, 128, 6, 8) -> (B, 128, 1, 1)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))

        # Head: 128 -> (n_waypoints * 2)
        self.fc    = nn.Linear(128, n_waypoints * 2)

        # (optional) tiny dropout for regularization
        self.drop  = nn.Dropout(p=0.1)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # image: (B, 3, 96, 128)
        x = F.relu(self.bn1(self.conv1(image)))   # (B, 32, 48, 64)
        x = F.relu(self.bn2(self.conv2(x)))       # (B, 64, 24, 32)
        x = F.relu(self.bn3(self.conv3(x)))       # (B,128, 12, 16)
        x = F.relu(self.bn4(self.conv4(x)))       # (B,128,  6,  8)

        x = self.gap(x)                           # (B,128,1,1)
        x = torch.flatten(x, 1)                   # (B,128)
        x = self.drop(x)                          # (B,128)

        x = self.fc(x)                            # (B, n_waypoints*2)
        out = x.view(x.size(0), self.n_waypoints, 2)  # (B, n_waypoints, 2)
        return out



























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
