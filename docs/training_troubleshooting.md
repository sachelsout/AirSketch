# Training Troubleshooting

## MPJPE not decreasing after 10 epochs (still > 40 px)

1. Learning rate too high.
   Try: --override training.learning_rate=0.0001

2. Batch contains NaN.
   Check:
     batch = next(iter(train_loader))
     print(torch.any(torch.isnan(batch["sequence"])))  # must be False

3. Landmarks not in [0, 1].
   Check:
     print(batch["sequence"].min(), batch["sequence"].max())

4. All gesture labels are idle (0).
   Check:
     print(train_loader.dataset.gesture_distribution())

## Gesture accuracy stuck at ~50% after 5 epochs

1. Class imbalance not handled.
   Confirm get_class_weights() was passed to AirSketchLoss.
   Without it, a model predicting all-idle hits ~80% and never learns draw.

2. Lambda too high (above 1.0 drowns the regression term).
   Check: configs/default.yaml model.gesture_loss_weight should be 0.5.

## CUDA out of memory

1. Reduce batch size:  --override training.batch_size=32
2. Reduce hidden dim:  --override model.hidden_dim=64
3. Check other processes: nvidia-smi --query-compute-apps=pid,used_memory --format=csv

## Loss is NaN from epoch 1

1. Gradient explosion -- confirm clip_grad_norm_ is in the loop (max_norm=1.0).
2. Try from scratch without --resume.
3. Learning rate above 0.01 can saturate the Sigmoid in the regression head.

## Checkpoint not saving

Check write permissions. On Zaratan, write to /scratch/$USER/ not ~/home/.
Home dir has a 20 GB quota that fills quickly with .pt files.