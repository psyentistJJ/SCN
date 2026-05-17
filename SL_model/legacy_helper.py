from helper import sm_simulation, entrainment_benchmark
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import json
from datetime import datetime
from tqdm import tqdm
import hashlib
from pprint import pprint
import pickle

def SCN_entrain_py(
        N, 
        D,
        lam, 
        gamma, 
        mean_period, 
        std_period, 
        W_prob, 
        W_mu,
        W_std, 
        sweep: bool,
        std_phase_noise: float = 0.05, # 1hr deviation after 24hr
        std_m_noise: float = 0.2,
        dt = 0.1,
        init_seed = 42,
        period_seed = 142,
        W_seed = 242,
        m_noise_seed = 342,
        phase_noise_seed = 442
):
    
    np.random.seed(init_seed)
    light = False


    T = 24 * D
    timesteps = np.arange(0, T, dt)                  # Number of integration steps                 
    n_timestep = len(timesteps)

    # Initialize state vectors z
    amplitudes = np.random.uniform(0.01, 0.1, N)
    phases = np.random.uniform(-np.pi, np.pi, N)
    z = amplitudes * np.exp(1j*phases)

    z_history = np.zeros((N, n_timestep), dtype=complex)


    #=========================================#
    #             Set period_seed             #
    #=========================================#
    np.random.seed(period_seed)

    periods = np.random.normal(loc=mean_period, scale=std_period, size=N)

    if np.any(periods <= 0):
        raise ValueError(
            f"FATAL: Sampled a negative or zero period! (Min: {periods.min():.2f}h). "
            "This will break the frequency calculation."
        )
    
    omegas = 2.0 * np.pi / periods


    
    #=========================================#
    #               Set W_seed                #
    #=========================================#
    np.random.seed(W_seed)
    # Connectome W (N x N)
    topology_mask = np.random.rand(N, N) <= W_prob
    np.fill_diagonal(topology_mask, 0)
    W = np.random.normal(W_mu, W_std, (N,N)) * topology_mask
    W = W.astype(np.complex128)

    #==========================================#
    #              Set m_noise_seed            #   
    #==========================================#
    np.random.seed(m_noise_seed)

    # External Input Vector I_ext (N,)
    # Form: constant drive pulling East (Real axis)
    s, m = sm_simulation(D=D, eta=std_m_noise, dt=dt)

    I_ext = np.zeros(N, dtype=complex)


    #==========================================#
    #            Set phase_noise_seed          #
    #==========================================#
    np.random.seed(phase_noise_seed)
    # Brownian noise for phases
    random_kicks = np.random.normal(size=(N,n_timestep))
    phase_noise = std_phase_noise * random_kicks

    linear_rate = np.exp((lam + 1j * omegas)*dt)
    N_syn = N * W_prob
    freerun_warmup = int((24*2)/dt)

    # 2. FULLY VECTORIZED INTEGRATION LOOP
    for t in range(n_timestep):

        # Store state
        z_history[:, t] = z

        if t == (n_timestep - 1):  # last iter is waste! break!
            break
        
        # --- Deterministic Drift ---
        coupling = (1.0 / N_syn) * (W @ z) * dt
        nonlinear = (1.0 + 1j * gamma) * (np.abs(z)**2) * z * dt
    
        z = z * linear_rate - nonlinear + coupling

        # --- Brownian Diffusion ---
        diffusion = np.exp(1j * phase_noise[:, t] * np.sqrt(dt))
        
        if t < freerun_warmup:
            z = z * diffusion

        elif t >= freerun_warmup:
            z_hypothetical = z * diffusion

            if not light:
                rotation = np.exp(1j * np.pi/2)
        
                rotated_history = np.mean(z_history[:, t]) * rotation
                rotated_hypo = np.mean(z_hypothetical) * rotation

                if np.angle(rotated_history) < 0 and np.angle(rotated_hypo) >= 0:
                    light=True
                    light_onset=t

                else:
                    z = z_hypothetical

            if light:
                I_ext[:] = m[t-light_onset] + 0.0j 
                z = (z + I_ext*dt) * diffusion

    phase_only_vectors = np.exp(1j *np.angle(z_history)) # discard amplitude
    R_phase = np.abs(np.mean(phase_only_vectors, axis=0)) 

    # ==========================================
    # 3. Packaging Simulation Results
    # ==========================================
    if not sweep:
        verbose = True

        entrainment_benchmark(z_history, s, timesteps, dt, light_onset, warmup_days=2, verbose=verbose)
        return {'timesteps': timesteps,
                'dt':dt,
                'z_history': z_history,
                'omegas': omegas,
                'R_phase': R_phase,
                's': s,
                'm': m,
                'light_onset': light_onset
                }
        
    
    if sweep:
        verbose = False

        plv, phase_rmse, period_error, phase_coherence = entrainment_benchmark(
        z_history, s, timesteps, dt, light_onset, warmup_days=2, verbose=verbose
    )
    

        return {'plv': plv,
                'phase_rmse': phase_rmse,
                'period_error': period_error,
                'phase_coherence': phase_coherence,
                }



def run_1D_sweep_py(
        base_config: dict,
        sweep_param: str, 
        sweep_range: np.ndarray, 
        seed_pack: dict | None = None
        ):
    

    sweep = True

    background_config = base_config.copy()
    background_config['sweep'] = sweep

    background_config.pop(sweep_param, None)

    if seed_pack is None:
        seed_pack = {
            'init_seed': 42,
            'period_seed': 142,
            'W_seed': 242,
            'm_noise_seed': 342,
            'phase_noise_seed': 442
        }

    for key in seed_pack.keys():
        background_config.pop(key, None)

    sweep_range_list = list(sweep_range)

    experiment_identity = {
        'sweep_param': sweep_param,
        'sweep_range': sweep_range_list,
        **seed_pack,
        'background_config': background_config,
    }
    # We convert it to a string and hash it. 
    # (sort_keys=True ensures the hash is identical even if dictionary order changes)
    # (default=str prevents crashes if you accidentally leave a numpy float in your config)
    encoded_dict = json.dumps(experiment_identity, sort_keys=True, default=str).encode()
    run_hash = hashlib.md5(encoded_dict).hexdigest()[:8]


    saving_dir = "output data/1D sweep"
    os.makedirs(saving_dir, exist_ok=True)
    filepath = f"{saving_dir}/1D_{sweep_param}_{run_hash}.pkl"

    if os.path.exists(filepath):
        print(f"You have run this config before. Directly loading from disk...")
        with open(filepath, 'rb') as f:
            return pickle.load(f)


    result_list = []
    for x in tqdm(sweep_range):

        results = SCN_entrain_py(**{**base_config, 
                                'sweep': sweep, 
                                 sweep_param: x,
                                 **seed_pack,
                                })

        results[sweep_param] = x
        result_list.append(results)
    
    df = pd.DataFrame(result_list)

    experiment_identity['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # Package everything
    payload = {
        'metadata': experiment_identity,
        'data': df,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(payload, f)
        
    print(f"Secured Data + Metadata in {filepath}")

    # ==========================================
    # 6. Update the Master Ledger (Log EVERYTHING)
    # ==========================================

    ledger_path = f"{saving_dir}/_MASTER_LEDGER.csv"
    
    # Create a nice string like "0.0 to 1.0 (50 steps)" for the CSV
    range_max = sweep_range[-1]
    range_min = sweep_range[0]
    range_gap = range_max - range_min
    range_str = f"{range_min :.3f} to {range_max :.3f} ({len(sweep_range)} steps)"

    # We create the base entry...
    ledger_entry = {
        'timestamp': experiment_identity['timestamp'],
        'hash_id': run_hash,
        'sweep_param': sweep_param,
        'sweep_range': range_str,
        **seed_pack,
        **background_config,
    }
    
    new_row_df = pd.DataFrame([ledger_entry])
    
    # Safe Saving Protocol (Handles Evolving Models)
    if os.path.exists(ledger_path):
        # We read the old ledger, concatenate the new row, and overwrite.
        # Why? Because if you add a new parameter to your model tomorrow, 
        # Pandas will automatically create a new column for it without crashing!
        existing_ledger = pd.read_csv(ledger_path)
        updated_ledger = pd.concat([existing_ledger, new_row_df], ignore_index=True)
        updated_ledger.to_csv(ledger_path, index=False)
    else:
        new_row_df.to_csv(ledger_path, index=False)

    return payload


def run_2D_sweep_py(base_config, 
                 param1_name, 
                 param1_range, 
                 param2_name, 
                 param2_range, 
                 seed_pack: dict | None = None):
    """
    Runs a 2D parameter sweep, caches the result to disk using a unique hash, 
    and returns a structured payload.
    """

    sweep = True
    
    # 1. Create the Background Config (Remove the sweep parameters)
    background_config = base_config.copy()
    
    background_config.pop(param1_name, None)
    background_config.pop(param2_name, None)

    if seed_pack is None:
        seed_pack = {
            'init_seed': 42,
            'period_seed': 142,
            'W_seed': 242,
            'm_noise_seed': 342,
            'phase_noise_seed': 442
        }

    for key in seed_pack.keys():
        background_config.pop(key, None)

    # 2. Build the Experiment Identity (Metadata)
    experiment_identity = {
        'param1_name': param1_name,
        'param1_range': list(param1_range),
        'param2_name': param2_name,
        'param2_range': list(param2_range),
        **seed_pack,
        'background_config': background_config
    }
    
    # 3. Create the Unique Fingerprint (Hash)
    encoded_dict = json.dumps(experiment_identity, sort_keys=True, default=str).encode()
    run_hash = hashlib.md5(encoded_dict).hexdigest()[:8]

    # 4. Check for Cached File
    saving_dir = "output data/2D sweep"
    os.makedirs(saving_dir, exist_ok=True)
    filepath = f"{saving_dir}/2D_{param1_name}_{param2_name}_{run_hash}.pkl"

    if os.path.exists(filepath):
        print(f"You have run this 2D config before. Directly loading from disk...")
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    # 5. Run the Sweep
    results_list = []
    total_sims = len(param1_range) * len(param2_range)
    
    with tqdm(total=total_sims, desc=f"Sweeping {param1_name} x {param2_name}") as pbar:
        for p1 in param1_range:
            for p2 in param2_range:
                
                pbar.set_postfix({param1_name: np.round(p1, 3), param2_name: np.round(p2, 3)})
                
                # Update config and run
                sim_result = SCN_entrain_py(**{**base_config, 
                                            param1_name: p1, 
                                            param2_name: p2, 
                                            'sweep': sweep, 
                                            **seed_pack
                                            })
                
                # Extract metrics (Add or remove based on what SCN_freerun returns)
                sweep_data = {
                    param1_name: p1,
                    param2_name: p2,
                    'plv': sim_result.get('plv', np.nan),
                    'phase_rmse': sim_result.get('phase_rmse', np.nan),
                    'period_error': sim_result.get('period_error', np.nan),
                    'phase_coherence': sim_result.get('phase_coherence', np.nan),
                }
                
                results_list.append(sweep_data)
                pbar.update(1)
                
    # 6. Build DataFrame and add Timestamp
    df = pd.DataFrame(results_list)
    experiment_identity['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 7. Package the Payload
    payload = {
        'metadata': experiment_identity,
        'data': df,
    }
    
    # 8. Save to Disk and Return
    with open(filepath, 'wb') as f:
        pickle.dump(payload, f)

    # ... [After the pickle.dump block] ...
    print(f"Secured Data + Metadata in {filepath}")

    # ==========================================
    # 6. Update the 2D Master Ledger
    # ==========================================

    ledger_path = f"{saving_dir}/_MASTER_LEDGER.csv"
    
    # Create nice strings for the ranges
    r1_str = f"{param1_range[0] :.3f} to {param1_range[-1] :.3f} ({len(param1_range)} steps)"
    r2_str = f"{param2_range[0] :.3f} to {param2_range[-1] :.3f} ({len(param2_range)} steps)"

    # Base entry for 2D
    ledger_entry = {
        'timestamp': experiment_identity['timestamp'],
        'hash_id': run_hash,
        'param1_name': param1_name,
        'param1_range': r1_str,
        'param2_name': param2_name,
        'param2_range': r2_str,
        **seed_pack,
        **background_config,
    }
    
    new_row_df = pd.DataFrame([ledger_entry])
    
    # Safe Saving Protocol 
    if os.path.exists(ledger_path):
        existing_ledger = pd.read_csv(ledger_path)
        updated_ledger = pd.concat([existing_ledger, new_row_df], ignore_index=True)
        updated_ledger.to_csv(ledger_path, index=False)
    else:
        new_row_df.to_csv(ledger_path, index=False)

    return payload