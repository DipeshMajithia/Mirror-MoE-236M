#!/usr/bin/env python3
"""Turbo Light Configuration (The 'Resumed' 100M).

Optimized for resume training from step 1000 with lower LR.
"""

class TurboConfig:
    # --- Hardware Optimization ---
    BATCH_SIZE = 2          
    BLOCK_SIZE = 512        
    # OPTIMIZED: Accelerate training loop
    # Effective Batch: 2 * 16 = 32 (Standard)
    GRAD_ACCUM = 16         
    
    # --- Model Architecture ("Turbo" MoE - Speed & Capacity) ---
    USE_MOE = True          
    DIM = 512
    
    # 1. SPECIALISTS (High Capacity, Sparse Activation)
    N_EXPERTS = 16          
    HIDDEN_DIM = 256        # 16 * 256 = 4096 Total Capacity (~500M scale)
    ACTIVE_EXPERTS = 2      # Only 2 active (Fast!)
    
    # 2. THE ANCHOR (Lightweight Stability)
    # Reduced from 2048 -> 512. 
    # Just enough to keep gradients flowing, not enough to slow it down.
    SHARED_DIM = 512        
    
    N_LAYERS = 8            
    VOCAB_SIZE = 32000        

    # --- Training Curriculum ---
    # We are restarting from scratch (Step 0) or Step 1000? 
    # Recommendation: Start fresh or load the Dense weights if you can.
    
    PHASE_A_STEPS = 5000    
    PHASE_B_STEPS = 1500    
    PHASE_C_STEPS = 1500    
    PHASE_D_STEPS = 100 # Chat Tuning Phase (Reduced for Speed)
    
    # --- Hyperparameters ---
    # MoE needs a slightly higher LR to "wake up" the experts, 
    # but 2e-4 is safe. 1e-4 might be too slow for sparse updates.
    LEARNING_RATE = 2.0e-4  
    MIN_LR = 1.0e-5
    
    # --- Data Sources ---
    DATA_FILES_GENERAL = [
        "production/data/fineweb_edu_shard_00.txt",
        "production/data/fineweb_edu_shard_01.txt", 
        "production/data/fineweb_edu_shard_02.txt",
    ]
    
    DATA_FILES_SPECIALIZED = [
        "production/data/gold_clusters/code_instruct.jsonl",
        "production/data/gold_clusters/dolly_diverse.jsonl",
        "production/data/gold_clusters/orca_diverse.jsonl",
    ]

    DATA_FILES_CHAT = [
        "production/data/gold_clusters/dolly_diverse.jsonl",
        "production/data/gold_clusters/orca_diverse.jsonl",
    ]
    
    ALL_DATA_FILES = DATA_FILES_GENERAL + DATA_FILES_SPECIALIZED + DATA_FILES_CHAT

    # --- Checkpointing ---
    CHECKPOINT_DIR = "checkpoints_v2" # Using existing dir
    CHECKPOINT_EVERY = 500
    LOG_EVERY = 10
    VAL_EVERY = 500

def get_config():
    return TurboConfig()
