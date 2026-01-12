import matplotlib.pyplot as plt
import pysampled as ps
import h5py
import argparse
import numpy as np
from sklearn.decomposition import PCA


PARAMS = {
    'tremor': {
        'bandpass': (7, 17),  # Hz, bandpass filter for tremor detection
        'smooth_win': 0.5,  # s, smoothing window for tremor power
        'detrend_airpls_lambda': 4000,  # lambda for airPLS detrending
        'lowpass': 2.,  # Hz, lowpass cutoff frequency for tremor detection algorithm
        'power_slope_threshold': dict(triceps=1/6, biceps=5/48, palm=1/4),  # (m/s^2)^2/s, thresholds based on pilot data
        'min_time_between_events': 0.2,  # s, minimum time between a change in state for tremor detection
    },
    'arm_speed': {
        'lowpass': 10.,  # Hz, lowpass filter for arm speed
        'pos_lowpass': 5.,  # Hz, lowpass filter for position data
        'speed_th': 5.,  # cm/s, speed threshold for considering that there was movement
        'min_time_between_events_reach': 1.,  # s, minimum time between a change in state during reach events (for discarding sudden changes)
    },
    'emg': {
        'lowpass': 20,  # Hz, lowpass filter for EMG
        'highpass': 500,  # Hz, highpass filter for EMG
        'notch': 60,  # Hz, notch filter frequency for EMG
        'rms_win_size': 0.05,  # s, window size for RMS calculation
        'envelope_sr': 240,  # Hz, final sampling rate for EMG envelope
    }
}


def process_arm_pc1_from_mocap(mocap_array, sr, t, t_lims, mocap_locs):
    # Reshape mocap_array to (n_samples, n_markers*3)
    T, M, C = mocap_array.shape
    mocap_data = mocap_array.reshape(T, M * C)

    # Create a pysampled.Data object to take advantage of the processing methods
    mocap_sampled = ps.Data(mocap_data, sr=sr, t0=t[0])
    mocap_data = mocap_sampled.apply(lambda x: x - np.nanmean(x)).interpnan(kind='linear').lowpass(PARAMS['arm_speed']['pos_lowpass'])()
    mocap_data = mocap_data - np.mean(mocap_data, axis=0)

    # Get initial and final indices and cut array
    idx_i = np.argmin(np.abs(t - t_lims[:, 0]))
    idx_f = np.argmin(np.abs(t - t_lims[:, 1])) + 1
    mocap_data = mocap_data[idx_i:idx_f]

    # Perform PCA on the markers data to get the main axis of movement
    fit = PCA(n_components=1).fit(mocap_data)
    _arm_pc1 = np.squeeze(fit.transform(mocap_data))

    # Ensure that positive values in PC1 slope correspond to arm extension, and negative correspond to retraction
    # Do this by flipping the signal if it is negatively correlated with the projection of the vector from shoulder to forearm on the XY plane
    # When the value of this vector is increasing, it is an extension, and when the value of this vector is decreasing, it is a retraction
    idx_shoulder = mocap_locs.tolist().index('shoulder')
    if 'hand' in mocap_locs:
        idx_hand = mocap_locs.tolist().index('hand')
    else:
        idx_hand = mocap_locs.tolist().index('forearm')
    shoulder_marker = mocap_array[idx_i:idx_f, idx_shoulder]
    hand_marker = mocap_array[idx_i:idx_f, idx_hand]
    xy_metric = np.linalg.norm(hand_marker - shoulder_marker, axis=1)

    sign_mul = np.sign(np.ma.corrcoef(np.ma.masked_invalid(xy_metric), np.ma.masked_invalid(_arm_pc1))[0, 1])

    # Construct the the final pysampled.Data object
    arm_pc1 = ps.Data(sign_mul*_arm_pc1, sr=sr, t0=t[idx_i]).lowpass(PARAMS['arm_speed']['pos_lowpass'])
    return arm_pc1


def process_arm_speed_from_pc1(arm_pc1):
    # Get speed from arm_pc1
    get_speed = lambda arm_pc1: arm_pc1.apply(np.gradient, 1/arm_pc1.sr).lowpass(PARAMS['arm_speed']['lowpass']).apply(np.abs)
    arm_speed = get_speed(arm_pc1)

    return arm_speed


def get_reach_events_from_arm_pc1(arm_pc1: ps.Data, t_lims: np.ndarray):
    # Get velocity from arm_pc1
    get_velocity = lambda arm_pc1: arm_pc1.apply(np.gradient, 1/arm_pc1.sr).lowpass(PARAMS['arm_speed']['lowpass'])
    arm_velocity = get_velocity(arm_pc1)

    # Find crossings of speed threshold
    extension_start, extension_end = arm_velocity.find_crossings(
        th=PARAMS['arm_speed']['speed_th'],
        th_time=PARAMS['arm_speed']['min_time_between_events_reach']
    )
    retraction_end, retraction_start = arm_velocity.find_crossings(
        th=-PARAMS['arm_speed']['speed_th'],
        th_time=PARAMS['arm_speed']['min_time_between_events_reach']
    )

    # Filter events to only include those within the trial time limits
    t_min, t_max = t_lims[0, 0], t_lims[0, 1]
    
    # Convert to numpy arrays
    extension_start = np.array(extension_start)
    extension_end = np.array(extension_end)
    retraction_start = np.array(retraction_start)
    retraction_end = np.array(retraction_end)

    # Filter events
    ext_mask = (extension_start >= t_min) & (extension_end <= t_max)
    extension_start = extension_start[ext_mask]
    extension_end = extension_end[ext_mask]
    
    ret_mask = (retraction_start >= t_min) & (retraction_end <= t_max)
    retraction_start = retraction_start[ret_mask]
    retraction_end = retraction_end[ret_mask]

    return (extension_start, extension_end), (retraction_start, retraction_end)


def process_tremor_from_acc(acc_array, loc_idx, sr, t0):
    acc_array = acc_array[:, loc_idx]
    acc_sampled = ps.Data(acc_array, sr=sr, t0=t0)
    tremor = acc_sampled\
        .bandpass(*PARAMS['tremor']['bandpass'])\
        .envelope()\
        .magnitude()\
        .detrend_airPLS(lambda_=PARAMS['tremor']['detrend_airpls_lambda'])\
        .apply(lambda x: x**2)\
        .smooth(PARAMS['tremor']['smooth_win'])
    return tremor[1:len(acc_sampled)]


def get_tremor_events(tremor_power: ps.Data, loc:str, t_lims: np.ndarray):
    sig = tremor_power.lowpass(PARAMS['tremor']['lowpass']).apply(np.gradient, 1/tremor_power.sr)
    tremor_start, _ = sig.find_crossings(
        th=PARAMS['tremor']['power_slope_threshold'][loc], 
        th_time=PARAMS['tremor']['min_time_between_events']
        )
    tremor_end, _ = sig.find_crossings(
        th=-PARAMS['tremor']['power_slope_threshold'][loc], 
        th_time=PARAMS['tremor']['min_time_between_events']
        )
    
    # Filter events to only include those within trial time limits
    t_min, t_max = t_lims[0, 0], t_lims[0, 1]
    
    # Convert to numpy arrays
    tremor_start = np.array(tremor_start)
    tremor_end = np.array(tremor_end)
    
    # Filter tremor events
    tremor_mask = (tremor_start >= t_min) & (tremor_start <= t_max)
    tremor_start = tremor_start[tremor_mask]
    tremor_mask = (tremor_end >= t_min) & (tremor_end <= t_max)
    tremor_end = tremor_end[tremor_mask]
    
    return tremor_start, tremor_end


def process_emg_from_raw(emg_array, loc_idx, sr, t0):
    rms = lambda x, ax: np.sqrt(np.mean(x**2))
    emg_array = emg_array[:, loc_idx]
    emg_sampled = ps.Data(emg_array, sr=sr, t0=t0)
    emg_amplitude = emg_sampled\
        .shift_baseline()\
        .highpass(PARAMS['emg']['lowpass'])\
        .lowpass(PARAMS['emg']['highpass'])\
        .notch(PARAMS['emg']['notch'])\
        .apply_running_win(rms, win_size=PARAMS['emg']['rms_win_size'], win_inc=1/PARAMS['emg']['envelope_sr'])
    return emg_amplitude


def main():
    parser = argparse.ArgumentParser(description="Process arm speed, tremor power, and EMG amplitude from raw data.")
    parser.add_argument('--input', type=str, required=True, help='Path to input HDF5 file containing raw data.')
    parser.add_argument('--tremor_location', type=str, required=False, default='triceps', help='Location of the accelerometer ("palm", "triceps", "biceps").')
    args = parser.parse_args()

    # Load the input HDF5 file relevant data
    with h5py.File(args.input, 'r') as f:
        # Data to process
        mocap_t = f['timeseries/mocap/time'][:]
        mocap_sr = f['timeseries/mocap'].attrs['sr']
        mocap_data = f['timeseries/mocap/data'][:]
        mocap_locs = f['timeseries/mocap'].attrs['markers']

        acc_t0 = f['timeseries/acc/time'][0]
        acc_sr = f['timeseries/acc'].attrs['sr']
        acc_data = f['timeseries/acc/data'][:]
        acc_locs = f['timeseries/acc'].attrs['locations']

        emg_t = f['timeseries/emg/time'][:]
        emg_sr = f['timeseries/emg'].attrs['sr']
        emg_data = f['timeseries/emg/data'][:]
        
        trial_lims = f['events/trial_limits'][:]

        # Data to compare with (already processed)
        processed_tremor = f['timeseries/tremor_power/data'][:]
        processed_tremor_t = f['timeseries/tremor_power/time'][:]
        processed_arm_speed = f['timeseries/arm_speed/data'][:]
        processed_arm_speed_t = f['timeseries/arm_speed/time'][:]
        processed_emg = f['timeseries/emg_amplitude/data'][:]
        processed_emg_t = f['timeseries/emg_amplitude/time'][:]

    # Process the data
    arm_pc1 = process_arm_pc1_from_mocap(mocap_data, sr=mocap_sr, t=mocap_t, t_lims=trial_lims, mocap_locs=mocap_locs)
    arm_speed = process_arm_speed_from_pc1(arm_pc1)
    (extension_start, extension_end), (retraction_start, retraction_end) = get_reach_events_from_arm_pc1(arm_pc1, t_lims=trial_lims)

    try:
        loc_idx = acc_locs.tolist().index(args.tremor_location)
    except ValueError as e:
        print(e)
        raise ValueError(f"Invalid tremor location '{args.tremor_location}'. Valid options are: {acc_locs}.")
    tremor = process_tremor_from_acc(acc_data, loc_idx=loc_idx, sr=acc_sr, t0=acc_t0)
    tremor_start, tremor_end = get_tremor_events(tremor, loc=args.tremor_location, t_lims=trial_lims)

    emg_amplitude = process_emg_from_raw(emg_data, loc_idx=loc_idx, sr=emg_sr, t0=emg_t[0])

    # Plot the results for comparison
    _, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    
    # Arm speed plot with extension/retraction events
    axs[0].plot(processed_arm_speed_t, processed_arm_speed, lw=1.5, label='Processed (from HDF5)')
    axs[0].plot(arm_speed.t, arm_speed(), '--', lw=1.5, label='Processed (from script)')
    
    # Add vertical dashed lines for extension events (green)
    for i, (start, end) in enumerate(zip(extension_start, extension_end)):
        axs[0].axvline(start, color='green', linestyle='--', alpha=0.7, 
                      label='Extension Start' if i == 0 else "")
        axs[0].axvline(end, color='green', linestyle=':', alpha=0.7, 
                      label='Extension End' if i == 0 else "")
    
    # Add vertical dashed lines for retraction events (blue)
    for i, (start, end) in enumerate(zip(retraction_start, retraction_end)):
        axs[0].axvline(start, color='blue', linestyle='--', alpha=0.7, 
                      label='Retraction Start' if i == 0 else "")
        axs[0].axvline(end, color='blue', linestyle=':', alpha=0.7, 
                      label='Retraction End' if i == 0 else "")
    
    axs[0].set_title('Arm Speed from Mocap Data')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Speed (cm/s)')
    axs[0].legend()
    axs[0].set_xlim([87, 118])
    
    # Tremor plot with tremor events
    axs[1].plot(processed_tremor_t, processed_tremor[:, loc_idx], lw=1.5, label='Processed (from HDF5)')
    axs[1].plot(tremor.t, tremor(), '--', lw=1.5, label='Processed (from script)')
    
    # Add vertical dashed lines for tremor events (green)
    for i, (start, end) in enumerate(zip(tremor_start, tremor_end)):
        axs[1].axvline(start, color='green', linestyle='--', alpha=0.7, 
                      label='Tremor Start' if i == 0 else "")
        axs[1].axvline(end, color='green', linestyle=':', alpha=0.7, 
                      label='Tremor End' if i == 0 else "")
    
    axs[1].set_title('Tremor Power from Accelerometer Data')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Power (m^2/s^4)')
    axs[1].legend()
    axs[1].set_xlim([87, 118])

    # EMG amplitude plot
    axs[2].plot(processed_emg_t, processed_emg[:, loc_idx], lw=1.5, label='Processed (from HDF5)')
    axs[2].plot(emg_amplitude.t, emg_amplitude(), '--', lw=1.5, label='Processed (from script)')
    axs[2].set_title('EMG Amplitude')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude (mV)')
    axs[2].legend()
    axs[2].set_xlim([87, 118])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()