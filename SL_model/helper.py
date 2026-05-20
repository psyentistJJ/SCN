import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import pickle
import os
import json
import hashlib
from datetime import datetime
from pprint import pprint
import math
import glob
from numba import njit
from typing import Any

def sigmoid(x, alpha=5, theta=0.2):
    x=x-theta
    return 1/(1+np.exp(-alpha*x))

def hill(x, n=2, K=0.2):
    return (x**n) / (K**n + x**n)

def sm_simulation(D, sigma, rng_m_noise, dt=0.1):
    T = 24 * D
    timesteps = np.arange(0, T, dt) ; duration=len(timesteps)
    omega=(2*np.pi)/24

    s = np.zeros(duration)  # true latent phase
    m = np.zeros(duration)  # measurement
    gaussian_noise = rng_m_noise.normal(0, 1, size=duration)


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
                  focus_day: int| None = None,
                  autoshow: bool = True):

    N, duration =results['z_history'].shape
    dt = results['dt']
    D = int(duration/(24/dt))
    timesteps = results['timesteps']

    z_history = results['z_history']
    real_parts = np.real(z_history)
    imag_parts = np.imag(z_history)
    R_phase = results['R_phase']

    mean_field = np.mean(z_history, axis=0)
    
    mf_real = np.real(mean_field)
    mf_imag = np.imag(mean_field)
    mf_angle = np.angle(mean_field)
    mf_amplitude = np.abs(mean_field)


    ############################## main plotting #################################
    if N < 2:
        
        ind_angle = mf_angle
        ind_amplitude = mf_amplitude

        fig0, axs = plt.subplots(1, 2, figsize=(12,15))
        # plot (0,0)
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

        # plot(0,1)
        axs[1].plot(real_parts[0], imag_parts[0], c='red')
        axs[1].axhline(0, color='black', linewidth=0.5)
        axs[1].axvline(0, color='black', linewidth=0.5)
        axs[1].set_title("Dynamics")
        axs[1].set_xlabel("Real Part (Firing Rate Deviation)")
        axs[1].set_ylabel("Imaginary Part")
    
        axs[1].grid(True, linestyle='--', alpha=0.6)
        axs[1].set_aspect('equal')


        unwrapped_angle = np.unwrap(mf_angle)
        network_frequency, _ = np.polyfit(timesteps, unwrapped_angle, deg=1)
        emergent_period = (2.0 * np.pi) / np.abs(network_frequency)

        # row 2, plot 0
        fig1 = plt.figure(figsize=(12,5))
        plt.plot(timesteps/24, np.cos(ind_angle), label = 'phase')
        plt.plot(timesteps/24, ind_amplitude, label = 'certainty')
        plt.xticks(range(D+1))
        plt.grid(True, axis="x")
        plt.xlabel('Time')
        plt.ylabel('R/amplitude/certainty')
        plt.title(f'empirical period:{emergent_period :.2f}hr')
        plt.legend()  
        
        plt.tight_layout()
        plt.show()

        if autoshow==False:
            plt.close(fig0)
            plt.close(fig1)

        return fig0, fig1
    
    elif N>=2:

        ####### Adjust the figure size dynamically##########
        x_span = np.nanmax(real_parts[:min(5, N), :]) - np.nanmin(real_parts[:min(5, N), :])
        y_span = np.nanmax(imag_parts[:min(5, N), :]) - np.nanmin(imag_parts[:min(5, N), :])

        spread = max(x_span, y_span)

        # dynamically scale figure size
        base_width = 12
        base_height = 15

        scale = np.clip(spread / 2.5, 1.0, 2.5)

        fig0, axs = plt.subplots(1, 2, figsize=(base_width * scale, base_height * scale))
        
        #### plot (0,0) ####  
        steps_per_day = int(24 / dt)
        D_truncated = len(mf_angle) // steps_per_day

        psi_2d = mf_angle[:D_truncated * steps_per_day].reshape(D_truncated, steps_per_day)

        im=axs[0].imshow(
            psi_2d,
            aspect=0.66,
            origin='lower',
            extent=[0, 24, 0, D_truncated],
            cmap='grey',  # or any other colormap 
            interpolation= 'none'  # or 'none'
        )

        axs[0].invert_yaxis()
        axs[0].set_xlabel('Hour of Day')
        axs[0].set_ylabel('Day')
        #cbar = fig0.colorbar(im, ax=axs[0], shrink = 0.1)


        #### plot (0,1) #### 

        #pop-out
        if focus_day is not None:
            focus_slice_start = int(24/dt) * (focus_day-1)
            focus_slice_end = int(24/dt) * focus_day
            if focus_slice_start <1 :
                mf_real_background = mf_real[focus_slice_end:]
                mf_imag_background = mf_imag[focus_slice_end:]
            else:
                mf_real_background = np.concatenate((mf_real[:focus_slice_start], mf_real[focus_slice_end:]))
                mf_imag_background = np.concatenate((mf_imag[:focus_slice_start], mf_imag[focus_slice_end:]))

            axs[1].plot(mf_real_background, mf_imag_background, c='red', linewidth=1.5, alpha = 0.3)

            # ============================================================#  

            mf_real_focus = mf_real[ focus_slice_start : focus_slice_end ]
            mf_imag_focus = mf_imag[ focus_slice_start : focus_slice_end ]

            axs[1].plot(mf_real_focus, mf_imag_focus, c='black', linewidth = 1 ,zorder = 3)
            axs[1].scatter(mf_real_focus, mf_imag_focus, c='red', s = 8, zorder = 4)

            # start of focus day
            axs[1].scatter(
            mf_real_focus[0], mf_imag_focus[0],
            c='red', s=100, marker='o', label='start', zorder = 4, edgecolors='black'
            )

            # end of focus day
            axs[1].scatter(
                mf_real_focus[-1], mf_imag_focus[-1],
                c='black', s=100, marker='x', label='end', linewidth=2 , zorder = 5
            )

            axs[1].axhline(0, color='black', linewidth=0.5)
            axs[1].axvline(0, color='black', linewidth=0.5)
            axs[1].set_title("Populational Dynamics")
            axs[1].set_xlabel("Real Part (Firing Rate Deviation)")
            axs[1].set_ylabel("Imaginary Part")
            axs[1].grid(True, linestyle='--', alpha=0.6)
            axs[1].set_aspect('equal')

        #standard
        else:
            axs[1].plot(mf_real, mf_imag, c='red', linewidth=1.5)
            axs[1].axhline(0, color='black', linewidth=0.5)
            axs[1].axvline(0, color='black', linewidth=0.5)
            axs[1].set_title("Populational Dynamics")
            axs[1].set_xlabel("Real Part (Firing Rate Deviation)")
            axs[1].set_ylabel("Imaginary Part")
            axs[1].grid(True, linestyle='--', alpha=0.6)
            axs[1].set_aspect('equal')

        plt.legend()
        plt.tight_layout()



        fig1 = plt.figure(figsize=(12,5))

        light_onset = results['light_onset'] 
        s = results['s']    
        m = results['m']

        unwrapped_angle = np.unwrap(mf_angle[light_onset:])
        entrained_network_frequency, _ = np.polyfit(timesteps[light_onset:], unwrapped_angle, deg=1)
        emergent_period = (2.0 * np.pi) / np.abs(entrained_network_frequency)

        plt.plot(timesteps/24, np.cos(mf_angle), label = 'phase')
        plt.plot(timesteps[light_onset:]/24, np.cos(s[:duration-light_onset]), label = 'true phase')
        plt.plot(timesteps/24, R_phase)
        plt.xticks(range(D+1))
        plt.grid(True, axis="x")
        plt.axhline(y=0, c='black', linestyle='--')
        plt.axvline(x=timesteps[light_onset]/24, c='orange', linestyle=':', label='light onset')
        plt.xlabel('Time')
        plt.ylabel('cos(phase)')
        plt.title(f'empirical entrained period:{emergent_period :.2f}hr')
        plt.legend()  

        fig2, axs = plt.subplots(1, 2, figsize=(15,5))
        
        angular_diff_postlight_rad = np.angle(np.exp(1j*(mf_angle[light_onset:] - s[:duration-light_onset])))

        angular_diff_postlight_hour =  angular_diff_postlight_rad * (24.0 / (2 * np.pi))
        inst_freq = np.gradient(np.unwrap(mf_angle), dt)

        axs[0].plot(timesteps[light_onset:]/24, np.unwrap(angular_diff_postlight_hour))
        axs[0].set_title('Phase Error')
        axs[0].set_ylabel('Error (hr)')
        axs[0].set_xlabel('Time (Hr)')


        
        axs[1].plot(timesteps/24, inst_freq)
        axs[1].axvline(x=light_onset/(24/dt), linestyle=':', color= "orange")
        axs[1].set_title('Instantaneous Frequency')
        axs[1].set_ylabel('Frequency (1/hr)')
        axs[1].set_xlabel('Time (Hr)')
        
        plt.tight_layout()


        fig3 = plt.figure(figsize=(12,5))

        gs = GridSpec(1, 2, width_ratios=[5, 1], wspace=0.05)
        ax_main = fig3.add_subplot(gs[0])
        ax_yhist = fig3.add_subplot(gs[1], sharey=ax_main)

        true_light = phase2sun(s)
        ax_main.scatter(timesteps[light_onset:]/24, m[:duration-light_onset], label = 'm', color='limegreen' )
        ax_main.plot(timesteps[light_onset:]/24, true_light[:duration-light_onset], color='orange', linewidth='3')
        
        # collapsed y-distribution
        mask = true_light[:duration-light_onset] > 1e-8
        vals = m[:duration-light_onset][mask]
        ax_yhist.hist(
            vals,
            bins=40,
            orientation='horizontal',
            weights=np.ones_like(vals) / len(vals)*100,
            alpha=0.6
        )

        ax_yhist.set_xlabel("Proportion (%)")
        ax_yhist.tick_params(labelleft=False)
        ax_yhist.grid(False)


        if autoshow==False:
            plt.close(fig0)
            plt.close(fig1)
    
        else:
            plt.show()

        return fig0, fig1, fig2

def C_impl_entrain(
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
        ):
    # Pre-allocate memory
    z_history = np.zeros((N, n_timestep), dtype=np.complex128)
    z = z_init.copy()
    
    light = False
    rotation = np.exp(1j * (np.pi / 2.0))

    linear_rate = np.exp((lam + 1j * omegas)*dt)
    N_syn = max(1.0, N * W_prob) # prevent dividing by zero
    freerun_warmup = int((24*2)/dt)
    light_n_timestep = len(m)

    light_onset=0 ## Placeholder to trick numba!!!

    for t in range(n_timestep):
        z_history[:, t] = z
        
        if t == (n_timestep - 1) or (light == True and (t-light_onset) > (light_n_timestep-1)):  # last iter is waste! break!
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
                
                rotated_history = np.mean(z_history[:, t]) * rotation
                rotated_hypo = np.mean(z_hypothetical) * rotation
                
                if np.angle(rotated_history) < 0.0 and np.angle(rotated_hypo) >= 0.0:
                    light = True
                    light_onset = t
                else:
                    z = z_hypothetical
            
            if light:
                I_ext = m[t - light_onset] + 0.0j 
                z = (z + (I_ext * dt)) * phase_noise[:,t]
    

    exp_end = light_onset + light_n_timestep
    z_history = z_history[:, :exp_end]
                
    return z_history, light_onset

decorator_object = njit(fastmath=True)
C_impl_entrain_fastmath: Any = decorator_object(C_impl_entrain)

decorator_object = njit(fastmath=False)
C_impl_entrain_strictmath: Any = decorator_object(C_impl_entrain)

def SCN_entrain(
        N, 
        D, # number of days intended for entrainment (from light_onset to the end)
        lam, 
        gamma, 
        mean_period, 
        std_period, 
        W_prob, 
        W_mu, 
        W_std, 
        sweep: bool,
        fastmath: bool = False,
        single_sample: bool | None = None, 
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
    # 1. Initialization
    T = 24 * (D+3) # 2 freerun warmup days + 1-day flexibility for alignment
    timesteps = np.arange(0, T, dt)                 
    n_timestep = len(timesteps)

    #=========================================#
    #               Set init_seed             #
    #=========================================#
    rng_init = np.random.default_rng(init_seed)
    amplitudes = rng_init.uniform(0.01, 0.1, N)
    phases = rng_init.uniform(-np.pi, np.pi, N)
    z_init = amplitudes * np.exp(1j*phases)

    #=========================================#
    #             Set period_seed             #
    #=========================================#
    rng_period = np.random.default_rng(period_seed)
    periods = rng_period.normal(loc=mean_period, scale=std_period, size=N)

    if np.any(periods <= 0):
        raise ValueError(
            f"FATAL: Sampled a negative or zero period! (Min: {periods.min():.2f}h). "
            "This will break the frequency calculation."
        )
    
    omegas = 2.0 * np.pi / periods

    #=========================================#
    #               Set W_seed                #
    #=========================================#
    rng_W = np.random.default_rng(W_seed)
    # Connectome W (N x N)
    topology_mask = rng_W.random((N, N)) <= W_prob
    np.fill_diagonal(topology_mask, 0)
    W = rng_W.normal(W_mu, W_std, (N,N)) * topology_mask
    W = W.astype(np.complex128)

    #=========================================#
    #             Set m_noise_seed            #
    #=========================================#
    rng_m_noise = np.random.default_rng(m_noise_seed)
    s, m = sm_simulation(D=D, sigma=std_m_noise, rng_m_noise = rng_m_noise, dt=dt)


    #=========================================#
    #           Set phase_noise_seed          #
    #=========================================#
    rng_phase_noise = np.random.default_rng(phase_noise_seed)
    # Brownian noise for phases
    random_kicks = rng_phase_noise.normal(size=(N,n_timestep))
    phase_noise = np.exp(1j * std_phase_noise * random_kicks * np.sqrt(dt)).astype(np.complex128)

    #=========================================#
    #           Set syn_noise_seed          #
    #=========================================#
    rng_syn_noise = np.random.default_rng(syn_noise_seed)

    # Complex isotropic synaptic/input noise.
    # Scaling: sqrt(dt), because this is SDE-style additive noise.
    syn_noise = std_syn_noise * np.sqrt(dt) * (
        rng_syn_noise.normal(size=(N, n_timestep))
        + 1j * rng_syn_noise.normal(size=(N, n_timestep))
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
        
    exp_end = light_onset + len(m)
    timesteps = timesteps[:exp_end]

    # 3. METRICS & PACKAGING 
    phase_only_vectors = np.exp(1j * np.angle(z_history))
    R_phase = np.abs(np.mean(phase_only_vectors, axis=0)) 

    if not sweep:
        if single_sample == True:
            verbose = False
        else: 
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
        
    elif sweep:
        verbose = False
        mf_amp_avg , phase_coherence, entrainment_ratio, rmse  = entrainment_benchmark(
            z_history, s, timesteps, dt, light_onset, warmup_days=2, verbose=verbose)
    
        return {
            'mf_amp_avg': mf_amp_avg,  
            'phase_coherence': phase_coherence,
            'entrainment_ratio': entrainment_ratio,
            'rmse': rmse
            }


def entrainment_benchmark(
        z_history, 
        s, 
        timesteps, 
        dt, 
        light_onset, 
        warmup_days = 2, 
        verbose: bool = False,
        ):
    """
    Metrics are computed for the steady state portion during lights-on period.
    """
    if z_history.ndim != 2:
        raise ValueError("z_history must have shape (N, duration)")
    
    # 1. Extract data when light is on
    # The network experiences light starting from index `light_onset`
    warmup_steps = int(round(24 * warmup_days/dt))
    eval_start = light_onset + warmup_steps

    z_eval = z_history[:, eval_start:]
    s_eval = s[warmup_steps:]
    t_eval = timesteps[eval_start:]


    mf_eval = np.mean(z_eval, axis=0)
    mf_angle_eval = np.angle(mf_eval)


    network_frequency, _ = np.polyfit(t_eval, np.unwrap(mf_angle_eval), deg=1)
    network_period = (2 * np.pi) / network_frequency

    #####################################################################
    #                  --- Benchmark Calculations ---                   #
    #####################################################################

    ################ Raw meanfield amplitude: the network health #############
    mf_amp_avg = np.mean(np.abs(mf_eval))

    ########################## The Network Health ############################
    phase_only_vectors = np.exp(1j * np.angle(z_eval))
    phase_coherence = np.mean(np.abs(np.mean(phase_only_vectors, axis=0)))

    ################ Circular rmse (wobbles around the signal) ################
    phase_diffs = np.angle(np.exp(1j * (mf_angle_eval - s_eval)))
    rmse = np.sqrt(np.mean(phase_diffs**2))
    rmse = rmse * (24/(2*np.pi)) # from rad to hr

    ##################### Entrainment ratio (Cycle drift) ######################
    n_cycle_network = (np.unwrap(mf_angle_eval)[-1] - np.unwrap(mf_angle_eval)[0]) / (2 * np.pi)
    n_cycle_light = (np.unwrap(s_eval)[-1] - np.unwrap(s_eval)[0]) / (2 * np.pi)
    entrainment_ratio = n_cycle_network / n_cycle_light


    if verbose:
        print(f"network period: {network_period}")
        print("====== SCN TRACKING BENCHMARKS ======")
        print(f"1. Raw amplitude             : {mf_amp_avg :.4f}")
        print(f"2. R_phase                   : {phase_coherence:.4f}  (Ideal: 1.0  -> zero spread)")
        print(f"3. entrainment ratio         : {entrainment_ratio:.4f}  (Ideal: 1.0 -> cycle attendance rate )")
        print(f"4. Circular RMSE             : {rmse:.4f}  hr (Ideal: 0.0 -> no wobbling)")
        print("=====================================")

    return mf_amp_avg , phase_coherence, entrainment_ratio, rmse




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
        


def run_1D_ensemble(
        base_config: dict,
        sweep_param: str,
        sweep_range: np.ndarray,
        n_sample: int = 5,
        master_seed: int = 42
):
    """
    The Unified 'God-Mode' Engine. 
    Handles simulation, granular caching, and macroscopic aggregation in one pass.
    """
    sweep = True

    background_config = base_config.copy()
    background_config['sweep'] = sweep


    background_config.pop(sweep_param, None)

    legacy_seeds = ['init_seed', 'period_seed', 'W_seed', 'm_noise_seed', 'phase_noise_seed', 'syn_noise_seed']
    for key in legacy_seeds:
        background_config.pop(key, None)

    sweep_interval = np.round(sweep_range[1] - sweep_range[0], 5) if len(sweep_range) > 1 else 0.0

    # ==========================================
    # 1. ENSEMBLE IDENTITY (The Macro Cache)
    # ==========================================
    experiment_identity = {
        'sweep_param': sweep_param,
        'sweep_range': list(sweep_range),
        'n_sample': n_sample,
        'master_seed': master_seed,
        'background_config': background_config
    }
    encoded_dict = json.dumps(experiment_identity, sort_keys=True, default=str).encode()
    ens_hash = hashlib.md5(encoded_dict).hexdigest()[:8]
    
    # We create two directories now: one for the final results, one for the individual universes
    ens_dir = "output data/1D ensemble/ensembles"
    sample_dir = "output data/1D ensemble/samples"
    os.makedirs(ens_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    ens_filepath = f"{ens_dir}/ENS_1D_{sweep_param}_{ens_hash}.pkl"

    if os.path.exists(ens_filepath):
        print(f"Macro-state already computed! Loading cached ensemble: {ens_hash}")
        with open(ens_filepath, 'rb') as f:
            return pickle.load(f)

    # ==========================================
    # 2. THE SEED MANAGER & INCREMENTAL COMPUTE
    # ==========================================
    master_seq = np.random.SeedSequence(master_seed)
    streams = master_seq.spawn(n_sample * 6)
    
    all_raw_dfs = []
    idx = 0
    
    for i in range(n_sample):
        # Build the exact mathematical identity of THIS specific universe
        pack = {
            'init_seed': streams[idx].generate_state(1)[0],
            'period_seed': streams[idx+1].generate_state(1)[0],
            'W_seed': streams[idx+2].generate_state(1)[0],
            'm_noise_seed': streams[idx+3].generate_state(1)[0],
            'phase_noise_seed': streams[idx+4].generate_state(1)[0],
            'syn_noise_seed': streams[idx+5].generate_state(1)[0]
        }
        idx += 6

        # Create a unique hash just for this one universe
        sample_identity = {
            'sweep_param': sweep_param,
            'sweep_interval': sweep_interval,
            'seed_pack': pack,
            'background_config': background_config
        }
        sample_hash = hashlib.md5(json.dumps(sample_identity, sort_keys=True, default=str).encode()).hexdigest()[:8]
        sample_filepath = f"{sample_dir}/1D_sample_{sweep_param}_{sample_hash}.pkl"

        if os.path.exists(sample_filepath):
            with open(sample_filepath, 'rb') as f:
                existing_df = pickle.load(f)
            
            # NEW: find only the requested x-values missing from the cached sample
            existing_xs = existing_df[sweep_param].values
            missing_xs = [x for x in sweep_range if not any(np.isclose(x, existing_xs, atol=1e-5))]
            
            # NEW: exact cache hit branch
            if len(missing_xs) == 0:
                print(f"--- Sample {i+1}/{n_sample} [100% CACHED] ---")
                df = existing_df

            # NEW: partial cache hit branch
            else:
                print(f"--- Sample {i+1}/{n_sample} [PARTIAL: Computing {len(missing_xs)} new points] ---")
                result_list = []
                for x in tqdm(missing_xs, leave=False):
                    results = SCN_entrain(**{**background_config, sweep_param: x, **pack})
                    results[sweep_param] = x
                    result_list.append(results)
                
                # NEW: merge old cached points with newly computed points
                df = (
                    pd.concat([existing_df, pd.DataFrame(result_list)])
                    .sort_values(by=sweep_param)
                    .reset_index(drop=True)
                )

                # NEW: update sample cache after stitching
                with open(sample_filepath, 'wb') as f:
                    pickle.dump(df, f)

        else:
            # CHANGED: print label now explicitly marks this as a new sample
            print(f"--- Simulating Sample {i+1}/{n_sample} [NEW] ---")
            result_list = []
            for x in tqdm(sweep_range, leave=False):
                results = SCN_entrain(**{**background_config, sweep_param: x, **pack})
                results[sweep_param] = x
                result_list.append(results)
                
            df = pd.DataFrame(result_list)
            with open(sample_filepath, 'wb') as f:
                pickle.dump(df, f)

        # NEW: filter cached/stiched sample back down to today's requested sweep_range
        mask = df[sweep_param].apply(lambda x: any(np.isclose(x, sweep_range, atol=1e-5)))
        df_filtered = df[mask].copy()
        df_filtered['sample_ID'] = i
        all_raw_dfs.append(df_filtered)

    # ==========================================
    # 3. MACROSCOPIC AGGREGATION & SAVING
    # ==========================================
    massive_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
    metrics = [col for col in massive_raw_df.columns if col not in [sweep_param, 'sample_ID']]
    
    agg_df = massive_raw_df.groupby(sweep_param)[metrics].agg(['mean', 'std']).reset_index()
    agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_df.columns]

    full_raw_df = massive_raw_df

    payload = {
        'metadata': experiment_identity,
        'data': agg_df,
        'raw_data': full_raw_df,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(ens_filepath, 'wb') as f:
        pickle.dump(payload, f)
        
    print(f"\n ENSEMBLE COMPLETE. Macroscopic state secured in {ens_filepath}")

    # ==========================================
    # 4. MASTER LEDGER UPDATE
    # ==========================================
    ledger_path = f"{ens_dir}/_MASTER_LEDGER.csv"
    ledger_entry = {
        'timestamp': payload['timestamp'],
        'hash_id': ens_hash,
        'sweep_param': sweep_param,
        'n_sample': n_sample,
        'master_seed': master_seed,
        **background_config
    }
    
    new_row = pd.DataFrame([ledger_entry])
    if os.path.exists(ledger_path):
        pd.concat([pd.read_csv(ledger_path), new_row], ignore_index=True).to_csv(ledger_path, index=False)
    else:
        new_row.to_csv(ledger_path, index=False)

    return payload

def plotting_1D_ensemble(payload, mode='standard', n_overlay_samples=None):
    """
    Integrated Plotting Function for ALL benchmarks.
    mode: 'standard' (Mean + Shaded Std Dev) or 'overlay' (Individual samples + Mean)
    """

    df = payload['data']
    raw_df = payload['raw_data']

    meta = payload['metadata']
    sweep_param = meta['sweep_param']
    sweep_range = meta['sweep_range']
    n_sample = meta['n_sample']
    background_config = meta['background_config']

    interval = sweep_range[1] - sweep_range[0]

    print("========= CONFIGURATION =========")
    pprint(background_config, sort_dicts=False)
    print(f'{sweep_param}: ({sweep_range[0]:.3f}, {sweep_range[-1] + interval :.3f}, {interval})')
    print(f'sample size: {n_sample}')
    print("=================================\n")

    if mode == 'overlay' and n_overlay_samples is not None:
        selected_ids = sorted(raw_df['sample_ID'].unique())[:n_overlay_samples]
        raw_df = raw_df[raw_df['sample_ID'].isin(selected_ids)]
    
    # 1. Identify all benchmarks automatically
    benchmarks = [col.replace('_mean', '') for col in df.columns if col.endswith('_mean')]
    num_plots = len(benchmarks)
    
    # 2. Set up the dynamic grid (Works for both modes now)
    cols = 2
    rows = math.ceil(num_plots / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    
    if num_plots > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    # 3. Loop through every benchmark and plot
    for i, bm in enumerate(benchmarks):
        x = df[sweep_param]
        y_mean = df[f"{bm}_mean"]
        
        if mode == 'overlay':
            # Plot individual micro-trajectories (Spaghetti) for this specific benchmark
            if 'sample_ID' in raw_df.columns:
                for s_id in raw_df['sample_ID'].unique():
                    universe = raw_df[raw_df['sample_ID'] == s_id]
                    # We don't label individual seeds here to prevent massive legend bloat
                    axs[i].plot(universe[sweep_param], universe[bm], 
                             alpha=0.3, linewidth=1.2)
            
            # Plot the solid mean line over them
            axs[i].plot(x, y_mean, color='black', linewidth=2.5, label='Ensemble Mean')
            
        elif mode == 'standard':
            # Standard Mode (Mean + Shaded Std Dev)
            y_std = df[f"{bm}_std"]
            y_std_err = y_std/np.sqrt(n_sample)
            axs[i].plot(x, y_mean, label='Mean', color='blue', linewidth=2)
            axs[i].fill_between(x, y_mean - 1.96*y_std_err, y_mean + 1.96*y_std_err, alpha=0.2, color='blue', label='95% CI')
        
        else: 
            axs[i].plot(x, y_mean, label='Mean', color='blue', linewidth=2)
            
        # Common formatting
        axs[i].set_title(bm.upper(), fontweight='bold')
        axs[i].set_xlabel(sweep_param)
        axs[i].grid(True, alpha=0.3, linestyle='--')
        
        # Add the zero-line strictly for period_error to easily spot phase-locking
        if bm == 'period_error' and (y_mean.min() < 0 and y_mean.max() > 0):
            axs[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
        axs[i].legend()

    # 4. Hide empty subplots
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    # Super title based on mode

    plt.tight_layout()
    plt.show()


def plot_inspection_grid(payload, target_metric='period_error', xlim=None):
    raw_df = payload['raw_data']
    meta = payload['metadata']
    sweep_param = meta['sweep_param']
    
    sample_ids = raw_df['sample_ID'].unique()
    num_samples = len(sample_ids)
    
    cols = 3
    rows = math.ceil(num_samples / cols)
    
    # We remove sharey=True to allow individual auto-scaling
    fig, axs = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axs = axs.flatten() if num_samples > 1 else [axs]
    
    for i, s_id in enumerate(sample_ids):
        universe = raw_df[raw_df['sample_ID'] == s_id]
        
        # 1. Plot individual data
        axs[i].plot(universe[sweep_param], universe[target_metric], 
                 color='blue', alpha=0.8, linewidth=1.5, label=f'Seed {s_id}')
        
        if target_metric == 'period_error':
            axs[i].axhline(y=0, color='red', linestyle='--', alpha=0.4)
        
        # 2. DYNAMIC PADDING LOGIC (X and Y)
        if xlim is not None:
            x_min, x_max = xlim
            x_range = x_max - x_min
            x_pad = x_range * 0.05 # 5% breathing room
            
            # Apply padded X-limits
            axs[i].set_xlim(x_min - x_pad, x_max)
            
            # Find the data subset that is actually visible to calculate Y-padding
            visible_data = universe[(universe[sweep_param] >= x_min) & 
                                    (universe[sweep_param] <= x_max)]
            
            if not visible_data.empty:
                y_min = visible_data[target_metric].min()
                y_max = visible_data[target_metric].max()
                
                # Add a 10% Y-buffer so the line doesn't touch top/bottom
                y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.5
                axs[i].set_ylim(y_min - y_pad, y_max + y_pad)
        
        axs[i].set_title(f"Sample {s_id}", fontweight='bold')
        axs[i].grid(True, alpha=0.2, linestyle='--')
        axs[i].legend(loc='best', fontsize='small')

    fig.suptitle(f"Micro-Inspection: {target_metric.upper()}", 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Label outer axes
    for ax in axs[-cols:]:
        ax.set_xlabel(sweep_param)
    for ax in axs[::cols]:
        ax.set_ylabel(target_metric)

    for j in range(num_samples, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def plot_unwrapped_phase_walk(base_config, D, sweep_param, target_value, sample_id, master_seed=42):
    """
    Summons a specific sample at a specific parameter point to investigate time-series phase slips.
    """
    # 1. Reverse-engineer the exact seed pack for this specific sample_ID
    master_seq = np.random.SeedSequence(master_seed)
    # We spawn exactly enough streams to reach the requested sample
    streams = master_seq.spawn((sample_id + 1) * 6) 
    
    idx = sample_id * 6
    pack = {
        'init_seed': streams[idx].generate_state(1)[0],
        'period_seed': streams[idx+1].generate_state(1)[0],
        'W_seed': streams[idx+2].generate_state(1)[0],
        'm_noise_seed': streams[idx+3].generate_state(1)[0],
        'phase_noise_seed': streams[idx+4].generate_state(1)[0],
        'syn_noise_seed': streams[idx+5].generate_state(1)[0]
    }
    
    # 2. Prepare the config for a SINGLE run (sweep=False)
    sim_config = base_config.copy()
    sim_config['sweep'] = False
    sim_config['single_sample'] = True
    sim_config[sweep_param] = target_value
    sim_config['D'] = D
    
    # Strip any legacy seeds just in case
    for key in ['init_seed', 'period_seed', 'W_seed', 'm_noise_seed', 'phase_noise_seed', 'syn_noise_seed']:
        sim_config.pop(key, None)
        
    print(f"🔍 Simulating Sample {sample_id} at {sweep_param} = {target_value}...")
    
    # 3. Run the physics engine (Returns full z_history dictionary!)
    results = SCN_entrain(**sim_config, **pack)
    
    z_history = results['z_history']
    timesteps = results['timesteps']
    light_onset = results['light_onset']
    dt = results['dt']
    
    # ==========================================
    # 4. CALCULATE UNWRAPPED PHASE DIFFERENCE
    # ==========================================
    # A. Get the Mean Field (Kuramoto Order Parameter)
    Z_mean = np.mean(z_history, axis=0) 
    mean_phase = np.angle(Z_mean)
    
    # B. Unwrap it so it doesn't snap at +pi/-pi
    unwrapped_phase = np.unwrap(mean_phase)
    
    # C. Generate the ideal 24h reference clock phase
    ideal_omega = 2.0 * np.pi / 24.0
    reference_phase = ideal_omega * timesteps
    
    # D. Calculate the difference and convert to hours for intuition
    phase_diff_rad = unwrapped_phase - reference_phase
    
    # Zero out the phase difference at the moment the light turns on to see the drift clearly
    if 0 < light_onset < len(phase_diff_rad):
         phase_diff_rad -= phase_diff_rad[light_onset]
         
    phase_diff_hours = phase_diff_rad * (24.0 / (2 * np.pi))
    
    # ==========================================
    # 5. PLOT THE PHASE WALK
    # ==========================================
    plt.figure(figsize=(12, 5))
    
    # Plot the time axis in Days instead of raw timesteps
    days = timesteps / 24.0 
    plt.plot(days, phase_diff_hours, color='crimson', linewidth=2, label='Phase Drift')
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Perfect Entrainment (0h Diff)')
    plt.axvline(light_onset * dt / 24.0, color='orange', linestyle=':', label='Light Onset')
    
    plt.title(f"Micro-Time Investigation: Sample {sample_id} @ {sweep_param} = {target_value}", fontweight='bold')
    plt.xlabel("Time (Days)")
    plt.ylabel("Phase Difference (Hours)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plot_dynamics(results)
    return 

def plot_phase_walk(
        base_config,
        sample_id, 
        sweep_param, 
        target_values, 
        master_seed=42, 
        focus_day: int| None = None):
    """
    Summons a specific sample and simulates it across MULTIPLE parameter values, 
    plotting all phase walks on the same axes for direct comparison.
    """
    # 1. Reverse-engineer the exact seed pack for this specific sample_ID
    master_seq = np.random.SeedSequence(master_seed)
    streams = master_seq.spawn((sample_id + 1) * 6) 
    
    idx = sample_id * 6
    pack = {
        'init_seed': streams[idx].generate_state(1)[0],
        'period_seed': streams[idx+1].generate_state(1)[0],
        'W_seed': streams[idx+2].generate_state(1)[0],
        'm_noise_seed': streams[idx+3].generate_state(1)[0],
        'phase_noise_seed': streams[idx+4].generate_state(1)[0],
        'syn_noise_seed': streams[idx+5].generate_state(1)[0]
    }
    
    # 2. Setup the Plot and Colormap
    plt.figure(figsize=(12, 6))
    
    first_light_onset = None
    stored_dt = None
    
    print(f" Simulating Sample {sample_id} across {len(target_values)} values of {sweep_param}...")

    if len(target_values) > 1:
        # Create a color gradient (e.g., from blue to red) based on the target values
        norm = mcolors.Normalize(vmin=min(target_values), vmax=max(target_values))
        cmap = cm.viridis  # type: ignore
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # This is a matplotlib quirk required to make it work without an image
        
        # Add the colorbar to the current axes
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label(f"{sweep_param} Magnitude", rotation=270, labelpad=15, fontweight='bold')

    # 3. Loop through every requested value
    for val in target_values:
        sim_config = base_config.copy()
        sim_config['sweep'] = False
        sim_config['single_sample'] = True
        sim_config[sweep_param] = val
        
        for key in ['init_seed', 'period_seed', 'W_seed', 'm_noise_seed', 'phase_noise_seed', 'syn_noise_seed']:
            sim_config.pop(key, None)
            
        # Run the physics engine
        results = SCN_entrain(**sim_config, **pack)
        
        z_history = results['z_history']
        timesteps = results['timesteps']
        light_onset = results['light_onset']
        dt = results['dt']
        
        # Save the first light onset for plotting the vertical reference line later
        if first_light_onset is None:
            first_light_onset = light_onset
            stored_dt = dt
        
        # Calculate Unwrapped Phase
        Z_mean = np.mean(z_history, axis=0) 
        unwrapped_phase = np.unwrap(np.angle(Z_mean))
        
        ideal_omega = 2.0 * np.pi / 24.0
        reference_phase = ideal_omega * timesteps
        
        phase_diff_rad = unwrapped_phase - reference_phase
        
        # Zero out at light onset
        if 0 < light_onset < len(phase_diff_rad):
            phase_diff_rad -= phase_diff_rad[light_onset]
            
        phase_diff_hours = phase_diff_rad * (24.0 / (2 * np.pi))
        
        days = timesteps / 24.0 
        # Plot this specific line with its mapped color
        if len(target_values) > 1:
            plt.plot(days, phase_diff_hours, color=cmap(norm(val)), linewidth=1.5, alpha=0.8, label=f"{val:.2f}")
            
        
        elif len(target_values) == 1:
            plt.plot(days, phase_diff_hours, color='crimson', linewidth=2, label='Phase Drift')

    # 4. Formatting the unified plot
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='0h Diff (Perfect Lock)')
    
    if first_light_onset is not None:
        plt.axvline(first_light_onset * stored_dt / 24.0, color='orange', linestyle=':', label='Approx Light Onset')
    
    plt.title(f"Phase Drift Dynamics: Sample {sample_id} | Varying {sweep_param}", fontweight='bold', fontsize=14)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Phase Difference (Hours)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create a nice legend for the parameters
    #plt.legend(title=f"{sweep_param} values", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    if len(target_values) == 1:
        plot_dynamics(results, focus_day)

    return