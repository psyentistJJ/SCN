import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle
import os
import json
import hashlib
from datetime import datetime
from pprint import pprint
import math
import seaborn as sns
import glob
from numba import njit
from typing import Any

def sigmoid(x, alpha=5, theta=0.2):
    x=x-theta
    return 1/(1+np.exp(-alpha*x))

def hill(x, n=2, K=0.2):
    return (x**n) / (K**n + x**n)

def sm_simulation(D, sigma, dt=0.1):
    T = 24 * D
    timesteps = np.arange(0, T, dt) ; duration=len(timesteps)
    omega=(2*np.pi)/24

    s = np.zeros(duration)  # true latent phase
    m = np.zeros(duration)  # measurement
    gaussian_noise = np.random.normal(0, 1, size=duration)


    for t in range(duration):
        if t==0:
            s[0] = -np.pi/2
        else:
            s[t] = s[t-1] + omega * dt
            
        s[t] = np.mod(s[t]+np.pi, 2 * np.pi)-np.pi

        I = max(np.cos(s[t]), 0)
        m_noise = sigma * np.sqrt(I) * gaussian_noise[t]

        m[t] = I + m_noise

    return s, m

def phase2sun(s: np.ndarray):
    return (abs(np.cos(s))+np.cos(s))/2

def plot_dynamics(results: dict,
                      autoshow: bool = True
):

    N, duration =results['z_history'].shape
    dt = results['dt']
    D = int(duration/(24/dt))
    timesteps = results['timesteps']

    z_history = results['z_history']
    real_parts = np.real(z_history)
    imag_parts = np.imag(z_history)
    R_phase = results['R_phase']

    z_mean = np.mean(z_history, axis=0)
    
    ##### mf stands for mean field #####
    mf_real = np.real(z_mean)
    mf_imag = np.imag(z_mean)
    mf_angle = np.angle(z_mean)
    mf_amplitude = np.abs(z_mean)

    unwrapped_angle = np.unwrap(mf_angle)
    network_frequency, _ = np.polyfit(timesteps, unwrapped_angle, deg=1)
    emergent_period = (2.0 * np.pi) / np.abs(network_frequency)

    if N < 2:
            
        ind_angle = mf_angle
        ind_amplitude = mf_amplitude

        fig0, axs = plt.subplots(1, 2, figsize=(12,15))
        # plot (0)
        steps_per_day = int(24 / dt)
        psi_2d = ind_angle[:].reshape(D, steps_per_day)
        im=axs[0].imshow(
            psi_2d,
            aspect=0.66,
            origin='lower',
            extent=[0, 24, 0, D],
            cmap='grey',  # or any other colormap 
            interpolation= 'none')
        
        axs[0].invert_yaxis()
        axs[0].set_xlabel('Hour of Day')
        axs[0].set_ylabel('Day')
        cbar = fig0.colorbar(im, ax=axs[0], label='Phase Angle (rad)', shrink=0.3)

        # plot(1)
        axs[1].plot(real_parts[0], imag_parts[0], c='red')
        axs[1].axhline(0, color='black', linewidth=0.5)
        axs[1].axvline(0, color='black', linewidth=0.5)
        axs[1].set_title("Dynamics")
        axs[1].set_xlabel("Real Part (Firing Rate Deviation)")
        axs[1].set_ylabel("Imaginary Part")
    
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].set_aspect('equal')


        

        #### plot (3) #### 
        fig1 = plt.figure(figsize=(12,5))
        plt.plot(timesteps/24, np.cos(ind_angle), label = 'phase')
        plt.plot(timesteps/24, ind_amplitude, label = 'certainty')
        plt.xticks(range(D+1))
        plt.grid(True, axis="x")
        plt.xlabel('Time')
        plt.ylabel('R/amplitude/certainty')
        plt.title(f'empirical period:{emergent_period :.2f}hr')
        plt.legend()  
        
        plt.tight_layout();

        if autoshow==False:
            plt.close(fig0)
            plt.close(fig1)

        return fig0, fig1
    
    elif N>=2:

        fig0, axs = plt.subplots(1, 3, figsize=(12, 15))

        for i in range(5):
            axs[1].plot(real_parts[i, :], imag_parts[i, :], alpha=0.8)
        axs[1].axhline(0, color='black', linewidth=0.5)
        axs[1].axvline(0, color='black', linewidth=0.5)
        axs[1].set_title("Individual Dynamics")
        axs[1].set_xlabel("Real Part (Firing Rate Deviation)")
        axs[1].set_ylabel("Imaginary Part")
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].set_aspect('equal')


        axs[2].plot(mf_real, mf_imag, c='red', linewidth=1.5)
        axs[2].axhline(0, color='black', linewidth=0.5)
        axs[2].axvline(0, color='black', linewidth=0.5)
        axs[2].set_title("Populational Dynamics")
        axs[2].set_xlabel("Real Part (Firing Rate Deviation)")
        axs[2].set_ylabel("Imaginary Part")
        axs[2].grid(True, linestyle='--', alpha=0.6)
        axs[2].set_aspect('equal')


        steps_per_day = int(24 / dt)
        psi_2d = mf_angle[:].reshape(D, steps_per_day)

        im=axs[0].imshow(
            psi_2d,
            aspect=0.66,
            origin='lower',
            extent=[0, 24, 0, D],
            cmap='grey',  # or any other colormap 
            interpolation= 'none'  # or 'none'
        )

        axs[0].invert_yaxis()
        axs[0].set_xlabel('Hour of Day')
        axs[0].set_ylabel('Day')
        #cbar = fig0.colorbar(im, ax=axs[0], shrink = 0.1)
        plt.tight_layout()

        fig1, axs = plt.subplots(1, 2, figsize=(15, 5))
        light_onset = results['light_onset']
        s = results['s']
        angular_diff = np.angle(np.exp(1j*(mf_angle[light_onset:] - s[:duration-light_onset])))
        axs[0].plot(timesteps/24, mf_angle, label = 'prediction')
        axs[0].plot(timesteps[light_onset:]/24, s[:duration-light_onset], label = 'ground truth')
        axs[0].plot(timesteps[light_onset:]/24, np.unwrap(angular_diff), label = 'prediction error')
        axs[0].set_xticks(range(D+1))
        axs[0].grid(True, axis="x")
        axs[0].axhline(y=0, c='black', linestyle='--')
        axs[0].axvline(x=timesteps[light_onset]/24, c='black', linestyle='--')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('angle (rad)')
        axs[0].legend()

        axs[1].plot(timesteps/24, mf_amplitude, label = 'certainty')
        axs[1].plot(timesteps/24, R_phase, label = 'phase synchrony')
        axs[1].set_title('synchrony & certainty')
        axs[1].grid(True, axis="x")
        axs[1].set_xticks(range(D+1))
        axs[1].set_xlabel('Time')
        axs[1].legend()

        plt.tight_layout()

        if autoshow==False:
            plt.close(fig0)
            plt.close(fig1)
    
        else:
            plt.show()

        return fig0, fig1

def C_impl_entrain(N, 
              n_timestep, 
              dt, 
              lam, 
              gamma, 
              W_prob,
              omegas, 
              z_init, 
              W, 
              phase_noise,
              syn_noise,
              m
              ):
    # Pre-allocate memory
    z_history = np.zeros((N, n_timestep), dtype=np.complex128)
    z = z_init.copy()
    
    light = False
    rotation = np.exp(1j * (np.pi / 2.0))

    linear_rate = np.exp((lam + 1j * omegas)*dt)
    N_syn = max(1.0, N * W_prob) # prevent dividing by zero
    freerun_warmup = int((24*2)/dt)

    light_onset=0 ## Placeholder to trick numba!!!

    for t in range(n_timestep):
        z_history[:, t] = z
        
        if t == (n_timestep - 1):  # last iter is waste! break!
            break
        
        # --- Deterministic Drift ---
        coupling_push = (1.0 / N_syn) * (W @ z) * dt + syn_noise[:, t]
        nonlinearity = (1.0 + 1j * gamma) * (np.abs(z)**2) * z * dt
    
        z = z * linear_rate - nonlinearity + coupling_push
        
        
        if t < freerun_warmup:
            z = z * phase_noise[:,t]
        elif t >= freerun_warmup:
            z_hypothetical = z * phase_noise[:,t]
            
            if not light:
                # Numba optimizes these mean calculations incredibly well
                rotated_history = np.mean(z_history[:, t]) * rotation
                rotated_hypo = np.mean(z_hypothetical) * rotation
                
                if np.angle(rotated_history) < 0.0 and np.angle(rotated_hypo) >= 0.0:
                    light = True
                    light_onset = t
                else:
                    z = z_hypothetical
            
            if light:
                # Vectorized update mapped straight to memory
                I_ext = m[t - light_onset] + 0.0j 
                z = (z + (I_ext * dt)) * phase_noise[:,t]
                
    return z_history, light_onset

decorator_object = njit(fastmath=True)
C_impl_entrain_fastmath: Any = decorator_object(C_impl_entrain)

decorator_object = njit(fastmath=False)
C_impl_entrain_strictmath: Any = decorator_object(C_impl_entrain)

def SCN_entrain(
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
        fastmath: bool = False,
        std_phase_noise: float = 0.05,
        std_syn_noise: float = 0.0, 
        std_m_noise: float = 0.2, 
        dt = 0.1,
        init_seed = 42, 
        period_seed = 142, 
        W_seed = 242, 
        m_noise_seed = 342, 
        phase_noise_seed = 442,
        syn_noise_seed = 542
        ):
    # 1. SETUP & SEED QUARANTINE (Runs in Python)
    T = 24 * D
    timesteps = np.arange(0, T, dt)                 
    n_timestep = len(timesteps)

    #=========================================#
    #               Set init_seed             #
    #=========================================#
    np.random.seed(init_seed)
    amplitudes = np.random.uniform(0.01, 0.1, N)
    phases = np.random.uniform(-np.pi, np.pi, N)
    z_init = amplitudes * np.exp(1j*phases)

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

    #=========================================#
    #             Set m_noise_seed            #
    #=========================================#
    np.random.seed(m_noise_seed)
    s, m = sm_simulation(D=D, sigma=std_m_noise, dt=dt)


    #=========================================#
    #           Set phase_noise_seed          #
    #=========================================#
    np.random.seed(phase_noise_seed)
    # Brownian noise for phases
    random_kicks = np.random.normal(size=(N,n_timestep))
    phase_noise = np.exp(1j * std_phase_noise * random_kicks * np.sqrt(dt)).astype(np.complex128)

    #=========================================#
    #           Set syn_noise_seed          #
    #=========================================#
    np.random.seed(syn_noise_seed)

    # Complex isotropic synaptic/input noise.
    # Scaling: sqrt(dt), because this is SDE-style additive noise.
    syn_noise = std_syn_noise * np.sqrt(dt) * (
        np.random.normal(size=(N, n_timestep))
        + 1j * np.random.normal(size=(N, n_timestep))
    ) / np.sqrt(2)

    syn_noise = syn_noise.astype(np.complex128)


    # 2. Implement in C to boost speed
    # We pass the pre-generated arrays into the compiled loop

    if fastmath:
        z_history, light_onset = C_impl_entrain_fastmath(
                                        N, 
                                        n_timestep, 
                                        dt, 
                                        lam, 
                                        gamma, 
                                        W_prob,
                                        omegas, 
                                        z_init, 
                                        W, 
                                        phase_noise,
                                        syn_noise,
                                        m
                                        )
    else:
        z_history, light_onset = C_impl_entrain_strictmath(
                                        N, 
                                        n_timestep, 
                                        dt, 
                                        lam, 
                                        gamma, 
                                        W_prob,
                                        omegas, 
                                        z_init, 
                                        W, 
                                        phase_noise,
                                        syn_noise, 
                                        m
                                        )


    # 3. METRICS & PACKAGING (Runs in Python)
    phase_only_vectors = np.exp(1j * np.angle(z_history))
    R_phase = np.abs(np.mean(phase_only_vectors, axis=0)) 

    if not sweep:
        verbose = True
        entrainment_benchmark(z_history, s, timesteps, dt, light_onset, warmup_days=2, verbose=verbose)
        return {
            'timesteps': timesteps, 
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
            z_history, s, timesteps, dt, light_onset, warmup_days=2, verbose=verbose)
        
        return {
            'plv': plv, 
            'phase_rmse': phase_rmse, 
            'period_error': period_error, 
            'phase_coherence': phase_coherence
            }


def entrainment_benchmark(
        z_history, 
        s, 
        timesteps, 
        dt, 
        light_onset, 
        warmup_days = 2, 
        target_period = 24.0, 
        verbose: bool = False,
        ):
    """
    Evaluate how well the SCN population phase tracks the true latent phase.
    Metrics are computed for the steady state portion during lights-on period.
    """
    if z_history.ndim != 2:
        raise ValueError("z_history must have shape (N, duration)")
    
    # 1. Extract data when light is on
    # The network experiences light starting from index `light_onset`
    z_light = z_history[:, light_onset:]
    t_light = timesteps[light_onset:]
    
    # The external signal `s` starts its cycle at index 0 from the network's perspective
    light_duration = z_light.shape[1]
    s_light = s[:light_duration]

    # 2. Further extract steady-state 
    warmup_steps = int(round(24.0  * warmup_days / dt))
    if warmup_steps >= light_duration - 2:
        raise ValueError("warmup_days leaves too few samples after light_onset for benchmark evaluation")

    z_light_stdy = z_light[:, warmup_steps:]
    s_light_stdy = s_light[warmup_steps:]
    t_light_stdy = t_light[warmup_steps:]

    # --- Benchmark Calculations (Unchanged) ---
    mean_field = np.mean(z_light_stdy, axis=0)
    mf_angle = np.angle(mean_field)

    phase_diffs = np.angle(np.exp(1j * (mf_angle - s_light_stdy)))
    plv = np.abs(np.mean(np.exp(1j * phase_diffs)))

    rmse = np.sqrt(np.mean(phase_diffs**2))
    rmse = rmse/np.pi

    unwrapped_angle = np.unwrap(mf_angle)
    network_frequency, _ = np.polyfit(t_light_stdy, unwrapped_angle, deg=1)
    if np.abs(network_frequency) < 1e-9:
        network_period = np.nan
        period_error = np.nan
    else:
        network_period = (2.0 * np.pi) / np.abs(network_frequency)
        period_error = network_period-target_period
        
    phase_only_vectors = np.exp(1j * np.angle(z_light_stdy))
    phase_coherence = np.mean(np.abs(np.mean(phase_only_vectors, axis=0)))

    if verbose:
        print("====== SCN TRACKING BENCHMARKS ======")
        print(f"1. Phase-Locking Value (PLV) : {plv:.4f}  (Ideal: 1.0 -> Stable locking)")
        print(f"2. Circular RMSE             : {rmse:.4f} π (Ideal: 0.0 -> Perfect prediction)")
        print(f"3. Period Matching Error     : {period_error:.4f} hr (Ideal: 0.0 -> 24h entrainment)")
        print(f"4. Phase Coherence           : {phase_coherence:.4f}  (Ideal: 1.0 -> High precision/unity)")
        print("=====================================")

    return plv, rmse, period_error , phase_coherence

def run_1D_sweep(
        base_config: dict,
        sweep_param: str, 
        sweep_range: np.ndarray, 
        seed_pack: dict | None = None,
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
            'phase_noise_seed': 442,
            'syn_noise_seed': 542
        }

    for key in seed_pack.keys():
        background_config.pop(key, None)

    sweep_range_list = list(sweep_range)

    experiment_identity = {
        'sweep_param': sweep_param,
        'sweep_range': sweep_range_list,
        'seed_pack': seed_pack,
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

        results = SCN_entrain(**{**base_config, 
                                'sweep': sweep, 
                                 sweep_param: x,
                                 **seed_pack
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


def plotting_1D_sweep(payload):
    seed_pack={}

    df = payload['data']
    meta = payload['metadata']
    
    seed_pack = meta['seed_pack']
    sweep_param = meta['sweep_param']
    sweep_range = meta['sweep_range']
    background_config = meta['background_config']

    interval = sweep_range[1] - sweep_range[0]

    print("========= CONFIGURATION =========")
    pprint(background_config, sort_dicts=False)
    print(f'{sweep_param}: ({sweep_range[0]:.3f}, {sweep_range[-1] + interval :.3f}, {interval})')

    print(f'seeds: {[v for v in seed_pack.values()]}')
    print("=================================\n")

    # 1. Identify how many plots we actually need to draw
    benchmarks = [col for col in df.columns if col != sweep_param]
    num_plots = len(benchmarks)
    
    # 2. Set up the grid (2 columns, dynamic number of rows)
    cols = 2
    rows = math.ceil(num_plots / cols)
    
    # Adjust the figure size based on how many rows we have (e.g., 4 inches tall per row)
    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    
    # Flatten the axs array so it's easy to iterate through, even if it's a 2D grid
    # If there is only 1 row, axs might be 1D, so we ensure it's a flat list.
    if num_plots > 1:
        axs = axs.flatten()
    else:
        axs = [axs] # Handle the rare case of only 1 benchmark

    # 3. Loop through and plot
    for i, benchmark in enumerate(benchmarks):
        axs[i].plot(df[sweep_param], df[benchmark])
        axs[i].set_title(benchmark)
        axs[i].set_xlabel(sweep_param)
        axs[i].grid(True, alpha=0.4, linestyle='--')

        if benchmark=='period_error' and (df['period_error'].min() < 0 and df['period_error'].max() > 0):
            axs[i].axhline(y=0, color='black', linestyle='--')

    # 4. Hide any empty subplots (if num_plots is an odd number like 5, the 6th slot is blank)
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()



def run_2D_sweep(
        base_config, 
        param1_name, 
        param1_range, 
        param2_name, 
        param2_range, 
        seed_pack: dict | None = None
        ):
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
            'phase_noise_seed': 442,
            'syn_noise_seed': 542
        }

    for key in seed_pack.keys():
        background_config.pop(key, None)

    # 2. Build the Experiment Identity (Metadata)
    experiment_identity = {
        'param1_name': param1_name,
        'param1_range': list(param1_range),
        'param2_name': param2_name,
        'param2_range': list(param2_range),
        'seed_pack': seed_pack,
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
                sim_result = SCN_entrain(**{**base_config, 
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



def plotting_2D_sweep(payload):
    """
    Consumes a 2D sweep payload and generates a grid of heatmaps.
    (Swapped: p2 is the Y-axis/Rows, p1 is the X-axis/Columns)
    """
    seed_pack = {}
    df = payload['data']
    meta = payload['metadata']
    p1 = meta['param1_name']
    p2 = meta['param2_name']

    seed_pack = meta['seed_pack']

    background_config = meta['background_config']

    interval2 = meta["param2_range"][1] - meta["param2_range"][0]
    interval1 = meta["param1_range"][1] - meta["param1_range"][0]

    # ==========================================
    # 1. Print Metadata (Swapped display)
    # ==========================================
    print("========= CONFIGURATION =========")
    pprint(background_config, sort_dicts=False)
    print(f'{p2}: ({meta["param2_range"][0]:.3f}, {meta["param2_range"][-1]:.3f}, {interval2})')
    print(f'{p1}: ({meta["param1_range"][0]:.3f}, {meta["param1_range"][-1]:.3f}, {interval1})')
    print(f'seeds: {seed_pack}')
    print("=================================\n")

    # ==========================================
    # 2. Setup Dynamic Grid
    # ==========================================
    benchmarks = [col for col in df.columns if col not in [p1, p2]]
    num_plots = len(benchmarks)
    
    cols = 2
    rows = math.ceil(num_plots / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    
    if num_plots > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    # Determine tick frequency (SWAPPED: X is now p1, Y is now p2)
    x_ticks_freq = max(1, len(meta["param1_range"]) // 10)
    y_ticks_freq = max(1, len(meta["param2_range"]) // 10)

    # ==========================================
    # 3. Loop, Pivot, and Plot Heatmaps
    # ==========================================
    for i, benchmark in enumerate(benchmarks):
        
        # Reshape the flat dataframe: p2 is now Rows (index), p1 is Cols (columns)
        pivot_table = df.pivot(index=p2, columns=p1, values=benchmark)
        
        # Sort the index descending so the highest Y-value is physically at the top
        pivot_table.sort_index(ascending=False, inplace=True)
        
        # Draw the heatmap
        sns.heatmap(
            pivot_table, 
            ax=axs[i], 
            cmap='viridis',     
            xticklabels=x_ticks_freq,      
            yticklabels=y_ticks_freq
        )
        
        # Format the axes to truncate long floats (e.g., 0.3333333 -> 0.33)
        axs[i].set_xticklabels([f"{float(t.get_text()):.2f}" for t in axs[i].get_xticklabels()], rotation=45)
        axs[i].set_yticklabels([f"{float(t.get_text()):.2f}" for t in axs[i].get_yticklabels()], rotation=0)

        axs[i].set_title(benchmark, fontweight='bold')
        axs[i].set_ylabel(p2) # Y-axis is now p2
        axs[i].set_xlabel(p1) # X-axis is now p1

    # ==========================================
    # 4. Hide Empty Subplots
    # ==========================================
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()


def plotting_2D_sweep_spatial(payload, elev=50, azim=270):
    """
    Consumes a 2D sweep payload and generates a grid of 3D Surface Plots.
    (Swapped: p2 is the Y-axis/Depth, p1 is the X-axis/Width)
    """

    seed_pack = {}

    df = payload['data']
    meta = payload['metadata']
    p1 = meta['param1_name']
    p2 = meta['param2_name']

    seed_pack = meta['seed_pack']

    background_config = meta['background_config']

    # ==========================================
    # 1. Print Metadata (Swapped display)
    # ==========================================
    print("=== BACKGROUND CONFIGURATION ===")
    pprint(background_config, sort_dicts=False)
    print(f'Y-Axis ({p2}): {meta["param2_range"][0]:.3f} ~ {meta["param2_range"][-1]:.3f}')
    print(f'X-Axis ({p1}): {meta["param1_range"][0]:.3f} ~ {meta["param1_range"][-1]:.3f}')
    print(f'seeds: {seed_pack}')
    print("================================\n")

    # ==========================================
    # 2. Setup Dynamic Grid
    # ==========================================
    benchmarks = [col for col in df.columns if col not in [p1, p2]]
    num_plots = len(benchmarks)
    
    cols = 2
    rows = math.ceil(num_plots / cols)
    
    # Tell matplotlib that every subplot needs to be 3D
    fig, axs = plt.subplots(rows, cols, figsize=(14, 6 * rows), subplot_kw={'projection': '3d'})
    
    if num_plots > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    # ==========================================
    # 3. Loop, Pivot, and Plot
    # ==========================================
    for i, benchmark in enumerate(benchmarks):
        
        # SWAPPED: index is now p2 (Rows/Depth), columns is now p1 (Cols/Width)
        pivot_table = df.pivot(index=p2, columns=p1, values=benchmark)
        
        # Matplotlib 3D needs a "Meshgrid" (coordinate grid)
        # X maps to columns (p1), Y maps to index (p2)
        X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
        Z = pivot_table.values
        
        # Draw the 3D surface
        surf = axs[i].plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        
        # Add a color bar attached to the surface
        #fig.colorbar(surf, ax=axs[i], shrink=0.5, aspect=10, pad=0.1)

        # Label everything (SWAPPED: X is p1, Y is p2)
        axs[i].set_title(benchmark, fontweight='bold')
        axs[i].set_ylabel(p2) 
        axs[i].set_xlabel(p1) 

        # Optional: Adjust the initial camera angle (Elevation, Azimuth)
        axs[i].view_init(elev=elev, azim=azim)

    # ==========================================
    # 4. Hide Empty Subplots
    # ==========================================
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def load_pickles(hash_id, folder="output data/1D sweep"):
    # Search the folder for ANY file that ends with this hash
    search_pattern = f"{folder}/*_{hash_id}.pkl"
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"Could not find a sweep with hash: {hash_id}")
        
    filepath = str(matching_files[0])
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    

def print_sweep_tree(folder="output data/1D sweep"):
    """
    Parses the _MASTER_LEDGER.csv in the specified folder and prints a hierarchical 
    tree of all experiments, dynamically adapting to 1D or 2D sweeps.
    """
    try:
        ledger = pd.read_csv(f"{folder}/_MASTER_LEDGER.csv")
    except FileNotFoundError:
        print(f"Ledger not found in '{folder}'! Run at least one sweep first.")
        return
    
    # 1. Dynamically detect if this is a 1D or 2D ledger
    is_2D = 'param1_name' in ledger.columns

    if is_2D:
        meta_cols = ['timestamp', 'hash_id', 'param1_name', 'param1_range', 'param2_name', 'param2_range']
    else:
        meta_cols = ['timestamp', 'hash_id', 'sweep_param', 'sweep_range'] 
        
    # 2. Isolate the Background Physics (Everything that isn't metadata)
    bg_cols = [c for c in ledger.columns if c not in meta_cols]
    
    # 3. Helper to turn a row of physics parameters into a clean string
    def make_bg_string(row):
        valid_items = row[bg_cols].dropna().to_dict()
        return " | ".join([f"{k}={v}" for k, v in valid_items.items()])
    
    ledger['bg_config_str'] = ledger.apply(make_bg_string, axis=1)
    
    print(f"\n📁 EXPERIMENT DIRECTORY: {folder}\n")
    
    # ==========================================
    # 2D Sweep Logic
    # ==========================================
    if is_2D:
        # Layer 1: What two variables are we sweeping?
        for (p1_name, p2_name), branch1 in ledger.groupby(['param1_name', 'param2_name']):
            print(f"[SWEEPING 2D]: {p1_name} x {p2_name}")
            
            # Layer 2: How far did we sweep them?
            for (p1_range, p2_range), branch2 in branch1.groupby(['param1_range', 'param2_range']):
                print(f"   ├── [RANGES]: X({p1_range}) | Y({p2_range})")
                
                # Layer 3: What was the background environment?
                for bg_str, branch3 in branch2.groupby('bg_config_str'):
                    print(f"   │    ├── [FIXED CONFIG]: {bg_str}")
                    
                    # Layer 4: The specific runs
                    for _, row in branch3.iterrows():
                        print(f"   │    │    └── 📄 Hash: {row['hash_id']}  (Run: {row['timestamp']})")
            print("") 

    # ==========================================
    # 1D Sweep Logic
    # ==========================================
    else:
        # Layer 1: What variable are we sweeping?
        for sweep_param, branch1 in ledger.groupby('sweep_param'):
            print(f"[SWEEPING 1D]: {sweep_param}")
            
            # Layer 2: How far did we sweep it?
            for sweep_range, branch2 in branch1.groupby('sweep_range'):
                print(f"   ├── [RANGE]: {sweep_range}")
                
                # Layer 3: What was the background environment?
                for bg_str, branch3 in branch2.groupby('bg_config_str'):
                    print(f"   │    ├── [FIXED CONFIG]: {bg_str}")
                    
                    # Layer 4: The specific runs
                    for _, row in branch3.iterrows():
                        print(f"   │    │    └── 📄 Hash: {row['hash_id']}  (Run: {row['timestamp']})")
            print("") 
        
