"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")


import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import numpy as np
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model, load_model
from homework.datasets.road_dataset import load_data


def train_epoch(
        model: nn.Module,
        loader,
        optimizer,
        device,
        grad_clip=1.0,
        lat_budget=0.6, 
        long_budget=0.2,
):
    

    #training loop
    model.train()

    #initialize metrics
    total_valid = 0.0
    loss_sum = 0.0
    lat_mae_sum = 0.0
    long_mae_sum = 0.0

    for batch in loader:
        #pulling a batch of tensors from the loader, putting them on device
        left  = batch["track_left"].to(device)              # (B, 10, 2)
        right = batch["track_right"].to(device)             # (B, 10, 2)
        target= batch["waypoints"].to(device)               # (B, 3, 2)
        mask  = batch["waypoints_mask"].to(device).bool()   # (B, 3)

        #Forwards pass through the model
        #pred = model(left, right)                     # (B, 3, 2)
        pred = model(batch["image"].to(device))

        #Calculating loss using smoothL1 loss function defined later
        loss, logs = planner_loss_smoothl1(
            pred, target, mask=mask,
            lat_budget=lat_budget, long_budget=long_budget, beta=1.0
        )

        #Backpropogation and parameter update
        #set_to_none: instead of zeroing out the previous gradient tensor, it sets the vector to None
        #   This is more efficient for memory
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        #gradient clipping rescales gradient vector if the norm is greater than gradient_clip = 1
        #do this with AdamW to prevent exploding gradients
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        #updating parameters using gradients
        optimizer.step()

        #accumulating stats by batch
        valid = mask.sum().item()                     # scalar count in this batch
        total_valid += valid
        loss_sum     += loss.item() * valid
        lat_mae_sum  += logs["lat_mae"]  * valid
        long_mae_sum += logs["long_mae"] * valid

    #epoch averages
    denom = max(total_valid, 1.0)
    return {
        "loss":      loss_sum / denom,
        "lat_mae":   lat_mae_sum / denom,
        "long_mae":  long_mae_sum / denom,
        "valid_pts": int(total_valid),
    }

    





@torch.no_grad()
def validate_epoch(model, loader, device, lat_budget=0.6, long_budget=0.2,):
    """
    Expects batch keys:
      'track_left'     : (B, 10, 2)
      'track_right'    : (B, 10, 2)
      'waypoints'      : (B, 3, 2)
      'waypoints_mask' : (B, 3)
    Returns epoch-averaged stats (weighted by # of valid waypoints).
    """
    #validation loop
    model.eval()

    #initialize metrics
    total_valid = 0.0
    loss_sum = 0.0
    lat_mae_sum = 0.0
    long_mae_sum = 0.0


    for batch in loader:
        left   = batch["track_left"].to(device)          # (B, 10, 2)
        right  = batch["track_right"].to(device)         # (B, 10, 2)
        target = batch["waypoints"].to(device)           # (B, 3, 2)
        mask   = batch["waypoints_mask"].to(device).bool()  # (B, 3)

        #forwards pass
        #pred = model(left, right)                        # (B, 3, 2)
        pred = model(batch["image"].to(device))

        #calculating loss
        loss, logs = planner_loss_smoothl1(
            pred, target, mask=mask,
            lat_budget=lat_budget, long_budget=long_budget, beta=1.0
        )

        #storing metrics per batch
        valid = mask.sum().item()
        total_valid += valid
        loss_sum     += loss.item() * valid
        lat_mae_sum  += logs["lat_mae"]  * valid
        long_mae_sum += logs["long_mae"] * valid

    #storing epoch info
    denom = max(total_valid, 1.0)
    return {
        "loss":     loss_sum / denom,
        "lat_mae":  lat_mae_sum / denom,
        "long_mae": long_mae_sum / denom,
        "valid_pts": int(total_valid),
    }





def planner_loss_smoothl1(pred, target, mask=None, *, lat_budget=0.6, long_budget=0.2, beta=1.0):
    """
    pred, target: (B, 3, 2) in ego frame  ->  [:, :, 0]=lateral (x), [:, :, 1]=longitudinal (z)
    mask: (B, 3) bool, True = valid waypoint (unmasked)

    Per-COORDINATE loss (x and z separately), not per-point Euclidean.
    We normalize by budgets so longitudinal counts ~3x more than lateral.
    """

    # Inputs
    # pred:   (B, 3, 2)   # predicted waypoints [x(lat), z(long)]
    # target: (B, 3, 2)   # ground-truth waypoints
    # mask:   (B, 3)      # True where waypoint is valid (not padded)

    # coordinate errors, each size (B, 3)
    dx = pred[..., 0] - target[..., 0] 
    dz = pred[..., 1] - target[..., 1]      

    # normalize by grading budgets, each size (B, 3)
    #The 'gradient budgets' are defined by the grader. we have a tighter threshold for longitudinal error
    #   so we are going to weight the longitudinal error more
    #NOTE: Currently dz(long) is multiplied by 5 and dx(lat) is multiplied by 5 / 3, may want to change this to just 3 and 1
    dx_n = dx / lat_budget
    dz_n = dz / long_budget

    # elementwise SmoothL1 (no reduction so we can mask)
    #Using smoothL1 so that we can take the error with respect to each coordinate within the point
    #this helps because we are able to weight the loss to penalize longitudinal error more heavily than lateral
    #torch.zeros_like(dx_n) is the target. We want zero difference between predictions and actuals
    lat_loss  = F.smooth_l1_loss(dx_n, torch.zeros_like(dx_n), beta=beta, reduction="none")  # (B,3)
    long_loss = F.smooth_l1_loss(dz_n, torch.zeros_like(dz_n), beta=beta, reduction="none")  # (B,3)

    #aggrigating loss per waypoint
    per_wp = lat_loss + long_loss  # (B,3)


    #NOTE: Using clamp min to avoid division by zero. clamp min takes the greater value between m.sum() and 1.0
    if mask is not None:
        #changing the data type of mask to match per_wp so they can be multiplied
        m = mask.to(per_wp.dtype)                 # (B,3)
        #aggregating loss per batch without invalid values
        #loss is a scalar for backpropogation
        loss = (per_wp * m).sum() / m.sum().clamp_min(1.0)
        # metrics (unnormalized MAE, masked)
        lat_mae  = (dx.abs() * m).sum() / m.sum().clamp_min(1.0)
        long_mae = (dz.abs() * m).sum() / m.sum().clamp_min(1.0)
    else:
        loss = per_wp.mean()
        lat_mae  = dx.abs().mean()
        long_mae = dz.abs().mean()

    return loss, {"lat_mae": float(lat_mae), "long_mae": float(long_mae)}





def main():

    #config
    train_data_dir = "drive_data/train"
    val_data_dir = "drive_data/val"
    batch_size = 64
    num_epochs = 20
    seed = 2024
    lat_budget = 0.6
    long_budget = 0.5

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)


    #setting device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Device: Cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")


    #datasets and loaders
    #potentially add transforms
    #Using state only so that we dont load image data that we dont need. We are training only on the ground truth values.
        # --- data (state_only = no images) ---
    train_loader = load_data(
        train_data_dir, transform_pipeline="default",
        return_dataloader=True, num_workers=2,
        batch_size=batch_size, shuffle=True
    )
    val_loader = load_data(
        val_data_dir, transform_pipeline="default",
        return_dataloader=True, num_workers=2,
        batch_size=batch_size, shuffle=False  
    )


    #model and optimizer
    model = CNNPlanner().to(device)
    #loss_fn = planner_loss_smoothl1()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3,  weight_decay=1e-4)
    

    


    # before the loop
    best_val = float("inf")
    patience = 10          # epochs with no improvement allowed
    min_delta = 0.0       # require this much improvement to reset patience
    stale = 0

    #training and validation
    #using early stopping to prevent overfitting
    for epoch in range(num_epochs):
        train_stats = train_epoch(model, train_loader, optimizer, device, grad_clip=1.0, lat_budget=lat_budget, long_budget=long_budget)
        val_stats   = validate_epoch(model, val_loader, device, lat_budget=lat_budget, long_budget=long_budget)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train: loss {train_stats['loss']:.4f}, "
            f"lat {train_stats['lat_mae']:.4f}, long {train_stats['long_mae']:.4f} | "
            f"Val:   loss {val_stats['loss']:.4f}, "
            f"lat {val_stats['lat_mae']:.4f}, long {val_stats['long_mae']:.4f}"
        )

        #stopping if it validates well enough
        if val_stats["lat_mae"] < 0.6 and val_stats["long_mae"] < 0.2:
            save_model(model) 
            print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val:.4f})")
            break

        # early stopping on validation loss
        val_metric = val_stats["loss"]
        if val_metric < best_val - min_delta:
            best_val = val_metric
            stale = 0
            save_model(model) 
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val:.4f})")
                break





if __name__ == "__main__":
    main()