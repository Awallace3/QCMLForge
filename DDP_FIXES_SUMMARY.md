# DDP Implementation Fixes for APNet3_fused

## Summary
Fixed the Distributed Data Parallel (DDP) implementation in the APNet3_fused model to properly support multi-GPU and multi-CPU training.

## Files Modified

### 1. `/src/apnet_pt/AtomPairwiseModels/apnet3_fused.py`

#### Fix 1: DDP Wrapper Initialization (Lines 1518-1523)
**Problem:** DDP wrapper was missing critical parameters for proper GPU/CPU device handling.

**Before:**
```python
self.model = DDP(
    self.model,
)
```

**After:**
```python
self.model = DDP(
    self.model,
    device_ids=[rank] if rank_device != "cpu" else None,
    output_device=rank if rank_device != "cpu" else None,
    find_unused_parameters=True,
)
```

**Why:**
- `device_ids`: Specifies which GPU device to use for each process
- `output_device`: Specifies where to gather outputs in multi-GPU scenarios
- `find_unused_parameters=True`: Required because the model has frozen parameters in `dimer_prop_model` (lines 144-156)

#### Fix 2: Sampler Epoch Setting (Lines 1596-1597)
**Problem:** DistributedSampler was not having its epoch set, causing poor data shuffling across epochs.

**Added:**
```python
for epoch in range(n_epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    # ... rest of training loop
```

**Why:**
- DistributedSampler needs to know the current epoch to ensure different data ordering across epochs
- Without this, all epochs would use the same data ordering, reducing training effectiveness

#### Fix 3: Removed Redundant Device Transfer (Line 1574)
**Problem:** Model was being moved to device twice - once before DDP wrapping (line 1501) and once after (line 1574).

**Before:**
```python
self.model = self.model.to(rank_device)
```

**After:**
```python
# self.model = self.model.to(rank_device)  # Redundant: already moved to device at line 1501
```

**Why:**
- Model should only be moved to device once, before DDP wrapping
- Moving after DDP wrapping is inefficient and potentially problematic

## New File Created

### 2. `/ddp_train.sh`
A complete example script demonstrating how to use the fixed DDP training functionality.

**Features:**
- Automatically detects number of available GPUs
- Falls back to CPU training if no GPUs available
- Configurable hyperparameters
- Uses APNet3-fused model with proper atom type parameter models
- Includes helpful output messages

**Usage:**
```bash
# Edit configuration in the script if needed
vim ddp_train.sh

# Run training
./ddp_train.sh
```

## How DDP Works Now

### Multi-GPU Training (world_size > 1)
1. `torch.multiprocessing.spawn()` creates multiple processes (line 1916-1931)
2. Each process:
   - Initializes its process group (NCCL for CUDA, GLOO for CPU) in `__setup()` (lines 1224-1231)
   - Gets assigned to a specific GPU via `rank_device` (lines 1492-1495)
   - Wraps the model with DDP (lines 1518-1523)
   - Creates DistributedSampler for data partitioning (lines 1528-1540)
   - Trains on its partition of data with gradient synchronization
   - Properly sets epoch for data shuffling (lines 1596-1597)

### Single-GPU/CPU Training (world_size = 1)
- Falls back to `single_proc_train()` method (lines 1935-1946)
- No DDP overhead
- Standard PyTorch training loop

## Testing

The implementation should now work correctly for:
- ✅ Single GPU training
- ✅ Multi-GPU training (world_size automatically detected in train_models.py:134)
- ✅ CPU-only training (uses GLOO backend)
- ✅ Models with frozen parameters (dimer_prop_model)

## Key Points

1. **Frozen Parameters:** The `dimer_prop_model` has frozen parameters (lines 144-156), which is why `find_unused_parameters=True` is required in DDP

2. **Device Handling:** The code properly handles both GPU and CPU devices:
   - GPU: Uses NCCL backend with device_ids
   - CPU: Uses GLOO backend without device_ids

3. **Data Distribution:** DistributedSampler ensures each process gets a unique subset of data, and epoch setting ensures proper shuffling

4. **Model Saving:** Only rank 0 saves the model (line 1616-1631), using `unwrap_model()` to get the underlying model without DDP wrapper

## References

- DDP Best Practices: https://pytorch.org/docs/stable/notes/ddp.html
- DistributedSampler: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
- Multiprocessing: https://pytorch.org/docs/stable/multiprocessing.html
