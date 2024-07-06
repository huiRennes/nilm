"""Various common functions and parameters.

Copyright (c) 2022, 2023 Lindo St. Angel.
"""

import os
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

# Alternative aggregate standardization parameters used for all appliances.
# From Michele D’Incecco, et. al., "Transfer Learning for Non-Intrusive Load Monitoring"
ALT_AGGREGATE_MEAN = 522.0  # in Watts
ALT_AGGREGATE_STD = 814.0   # in Watts

# If True the alternative standardization parameters will be used
# for scaling the datasets.
USE_ALT_STANDARDIZATION = False

# If True the appliance dataset will be normalized to [0, max_on_power]
# else the appliance dataset will be z-score standardized.
USE_APPLIANCE_NORMALIZATION = True

# Power consumption sample update period in seconds.
SAMPLE_PERIOD = 8

# Various parameters used for training, validation and testing.
# Except where noted, values are calculated from statistical analysis
# of the respective dataset.
params_appliance = {
    'kettle': {
        # Input sample window length (samples).
        'window_length': 599,
        # Appliance considered inactive below this power draw (W).
        # From Zhenrui Yue, et. al., "BERT4NILM: A Bidirectional Transformer Model
        # for Non-Intrusive Load Monitoring".
        'on_power_threshold': 2000.0,
        # Appliance max power draw (W).
        # From Zhenrui Yue, et. al., "BERT4NILM: A Bidirectional Transformer Model
        # for Non-Intrusive Load Monitoring".
        'train_max_on_power': 3968,
        'test_max_on_power': 3584,
        'validation_max_on_power': 3968,
        # If appliance power draw exceeds 'on_power_threshold' for at least this
        # value, it will be be considered to be active ('on' status) (s).
        # From Zhenrui Yue, et. al., "BERT4NILM: A Bidirectional Transformer Model
        # for Non-Intrusive Load Monitoring".
        'min_on_duration': 12.0,
        # For purposes of determining if an appliance is inactive ('off' status),
        # ignore changes in appliance power if less than or equal to this value (s).
        # From Zhenrui Yue, et. al., "BERT4NILM: A Bidirectional Transformer Model
        # for Non-Intrusive Load Monitoring".
        'min_off_duration': 0.0,
        # Training aggregate dataset mean (W).
        'train_agg_mean': 564.0304687889887,
        # Training aggregate dataset standard deviation (W).
        'train_agg_std': 835.2925009671787,
        # Training appliance dataset mean (W).
        'train_app_mean': 2055.4838367293455,
        # Training appliance dataset standard deviation (W).
        'train_app_std': 970.0731362946979,
        # Test aggregate dataset mean (W)
        'test_agg_mean': 291.31571761132903,
        'test_agg_std': 380.57765719102224,
        # Test appliance dataset mean (W).
        'test_app_mean': 2447.0973363929475,
        'test_app_std': 1101.8827204224647,
        'validation_agg_mean': 377.9968796216541,
        'validation_agg_std': 469.7123584139814,
        'validation_app_mean': 2343.4051142621147,
        'validation_app_std': 1004.7597006475886,
        # Appliance dataset alternative standardization mean (W).
        # From Michele D’Incecco, et. al., "Transfer Learning for
        # Non-Intrusive Load Monitoring"
        'alt_app_mean': 700.0,
        # Appliance dataset alternative standardization std (W).
        # From Michele D’Incecco, et. al., "Transfer Learning for
        # Non-Intrusive Load Monitoring"
        'alt_app_std': 1000.0,
        # Coefficient 0 (L1 loss multiplier).
        # From Zhenrui Yue, et. al., "BERT4NILM: A Bidirectional Transformer Model
        # for Non-Intrusive Load Monitoring".
        'c0': 1.0
    },
    'microwave': {
        'window_length': 599,
        'on_power_threshold': 200.0,
        'train_max_on_power': 3778,
        'test_max_on_power': 3592,
        'validation_max_on_power': 2050,
        'min_on_duration': 12.0,
        'min_off_duration': 30.0,
        'train_agg_mean': 506.7278030164707,
        'train_agg_std': 766.9915084535828,
        'train_app_mean': 510.03960527944633,
        'train_app_std': 742.30648118481,
        'validation_agg_mean': 405.7487526594589,
        'validation_agg_std': 784.3991706115174,
        'validation_app_mean': 1233.5959764845477,
        'validation_app_std': 371.4518403669119,
        'test_agg_mean': 381.21947760009573,
        'test_agg_std': 428.3196100268874,
        'test_app_mean': 761.1351981014673,
        'test_app_std': 524.6032031645268,
        'alt_app_mean': 500.0,
        'alt_app_std': 800.0,
        'c0': 1.0
    },
    'fridge': {
        'window_length': 599,
        'on_power_threshold': 50.0,
        'train_max_on_power': 3584,
        'test_max_on_power': 2048,
        'validation_max_on_power': 3968,
        'min_on_duration': 60.0,
        'min_off_duration': 12.0,
        'train_agg_mean': 523.7190021344519,
        'train_agg_std': 855.2135152940822,
        'train_app_mean': 87.81715351268862,
        'train_app_std': 46.963790598276695,
        'test_app_mean': 77.61295826073543,
        'test_app_std': 26.08558479292136,
        'test_agg_mean': 254.83499472989803,
        'test_agg_std': 350.9031749257443,
        'validation_agg_mean': 367.2209206733284,
        'validation_agg_std': 618.0497897911025,
        'validation_app_mean': 84.61324975067673,
        'validation_app_std': 36.33300917160002,
        'alt_app_mean': 200.0,
        'alt_app_std': 400.0,
        'c0': 1e-06
    },
    'dishwasher': {
        'window_length': 599,
        'on_power_threshold': 700,
        'train_max_on_power': 3840.0,
        'test_max_on_power': 3588,
        'validation_max_on_power': 3488,
        'min_on_duration': 1800.0,
        'min_off_duration': 1800.0,
        'train_agg_mean': 573.4013850661521,
        'train_agg_std': 806.8727969742708,
        'train_app_mean': 882.8289567908794,
        'train_app_std': 973.6000844761928,
        'test_app_mean': 769.6329747872509,
        'test_app_std': 1024.2289464497765,
        'test_agg_mean': 377.99695178755263,
        'test_agg_std': 469.7123751996451,
        'validation_agg_mean': 451.7358885032802,
        'validation_agg_std': 501.96875450389484,
        'validation_app_mean': 1498.1607203092972,
        'validation_app_std': 1287.7188158444567,
        'alt_app_mean': 700.0,
        'alt_app_std': 1000.0,
        'c0': 1.0
    },
    'washingmachine': {
        'window_length': 599,
        'on_power_threshold': 400,
        'train_max_on_power': 3968,
        'test_max_on_power': 3972,
        'validation_max_on_power': 2475,
        'min_on_duration': 1800.0,
        'min_off_duration': 160.0,
        'train_agg_mean': 507.3718738840091,
        'train_agg_std': 762.3650210515159,
        'train_app_mean': 502.2916058612964,
        'train_app_std': 756.6312247017426,
        'test_app_mean': 976.6399670010271,
        'test_app_std': 1033.5390027476958,
        'test_agg_mean': 685.6207706598715,
        'test_agg_std': 1163.2520493471122,
        'validation_agg_mean': 451.7358885032802,
        'validation_agg_std': 501.96875450389484,
        'validation_app_mean': 467.3980720105909,
        'validation_app_std': 763.6700722422596,
        'alt_app_mean': 400.0,
        'alt_app_std': 700.0,
        'c0': 1e-02
    }
}

def find_test_filename(test_dir, appliance, test_type) -> str:
    """Find test file name given a datset name."""
    for filename in os.listdir(os.path.join(test_dir, appliance)):
        if test_type == 'train' and 'TRAIN' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'uk' and 'UK' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'redd' and 'REDD' in filename.upper():
            test_filename = filename
            break
        elif test_type == 'test' and 'TEST' in\
                filename.upper() and 'TRAIN' not in filename.upper() and 'UK' not in filename.upper():
            test_filename = filename
            break
        elif test_type == 'val' and 'VALIDATION' in filename.upper():
            test_filename = filename
            break
    return test_filename

def load_dataset(file_name, crop=None):
    """Load CSV file and return mains power, appliance power and status."""
    df = pd.read_csv(file_name, nrows=crop)

    mains_power = np.array(df.iloc[:, 0], dtype=np.float32)
    appliance_power = np.array(df.iloc[:, 1], dtype=np.float32)
    activations = np.array(df.iloc[:, 2], dtype=np.float32)

    return mains_power, appliance_power, activations

def tflite_infer(interpreter, provider, num_eval, eval_offset=0, log=print) -> list:
    """Perform inference using a tflite model"""
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    log(f'interpreter input details: {input_details}')
    output_details = interpreter.get_output_details()
    log(f'interpreter output details: {output_details}')
    # Check I/O tensor type.
    input_dtype = input_details[0]['dtype']
    floating_input = input_dtype == np.float32
    log(f'tflite model floating input: {floating_input}')
    output_dtype = output_details[0]['dtype']
    floating_output = output_dtype == np.float32
    log(f'tflite model floating output: {floating_output}')
    # Get I/O indices.
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    # If model has int I/O get quantization information.
    if not floating_input:
        input_quant_params = input_details[0]['quantization_parameters']
        input_scale = input_quant_params['scales'][0]
        input_zero_point = input_quant_params['zero_points'][0]
    if not floating_output:
        output_quant_params = output_details[0]['quantization_parameters']
        output_scale = output_quant_params['scales'][0]
        output_zero_point = output_quant_params['zero_points'][0]

    # Calculate num_eval sized indices of contiguous locations in provider.
    # Get number of samples per batch in provider. Since batch should always be
    # set to 1 for inference, this will simply return the total number of samples.
    samples_per_batch = len(provider)
    if num_eval - eval_offset > samples_per_batch:
        raise ValueError('Not enough test samples to run evaluation.')
    eval_indices = list(range(samples_per_batch))[eval_offset:num_eval+eval_offset]

    log(f'Running inference on {num_eval} samples...')
    start = time.time()
    def infer(i):
        sample, target, _= provider[i]
        if not sample.any():
            return 0.0, 0.0 # ignore missing data
        ground_truth = np.squeeze(target)
        if not floating_input: # convert float to int
            sample = sample / input_scale + input_zero_point
            sample = sample.astype(input_dtype)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke() # run inference
        result = interpreter.get_tensor(output_index)
        prediction = np.squeeze(result)
        if not floating_output: # convert int to float
            prediction = (prediction - output_zero_point) * output_scale
        #print(f'sample index: {i} ground_truth: {ground_truth:.3f} prediction: {prediction:.3f}')
        return ground_truth, prediction
    results = [infer(i) for i in tqdm(eval_indices)]
    end = time.time()
    log('Inference run complete.')
    log(f'Inference rate: {num_eval / (end - start):.3f} Hz')

    return results

def normalize(dataset):
    """Normalize or standardize a dataset."""
    # Compute aggregate statistics.
    agg_mean = np.mean(dataset[0])
    agg_std = np.std(dataset[0])
    print(f'agg mean: {agg_mean}, agg std: {agg_std}')
    agg_median = np.percentile(dataset[0], 50)
    agg_quartile1 = np.percentile(dataset[0], 25)
    agg_quartile3 = np.percentile(dataset[0], 75)
    print(f'agg median: {agg_median}, agg q1: {agg_quartile1}, agg q3: {agg_quartile3}')
    # Compute appliance statistics.
    app_mean = np.mean(dataset[1])
    app_std = np.std(dataset[1])
    print(f'app mean: {app_mean}, app std: {app_std}')
    app_median = np.percentile(dataset[1], 50)
    app_quartile1 = np.percentile(dataset[1], 25)
    app_quartile3 = np.percentile(dataset[1], 75)
    print(f'app median: {app_median}, app q1: {app_quartile1}, app q3: {app_quartile3}')
    def z_norm(dataset, mean, std):
        return (dataset - mean) / std
    def robust_scaler(dataset, median, quartile1, quartile3): #pylint: disable=unused-variable
        return (dataset - median) / (quartile3 - quartile1)
    return (
        z_norm(
            dataset[0], agg_mean, agg_std),
        z_norm(
            dataset[1], app_mean, app_std))

def compute_status(appliance_power:np.ndarray, appliance:str) -> list:
    """Compute appliance on-off status."""
    threshold = params_appliance[appliance]['on_power_threshold']

    def ceildiv(a:int, b:int) -> int:
        """Upside-down floor division."""
        return -(a // -b)

    # Convert durations from seconds to samples.
    min_on_duration = ceildiv(params_appliance[appliance]['min_on_duration'],
                              SAMPLE_PERIOD)
    min_off_duration = ceildiv(params_appliance[appliance]['min_off_duration'],
                               SAMPLE_PERIOD)

    # Apply threshold to appliance powers.
    initial_status = appliance_power.copy() >= threshold

    # Find transistion indices.
    status_diff = np.diff(initial_status)
    events_idx = status_diff.nonzero()
    events_idx = np.array(events_idx).squeeze()
    events_idx += 1

    # Adjustment for first and last transition.
    if initial_status[0]:
        events_idx = np.insert(events_idx, 0, 0)
    if initial_status[-1]:
        events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

    # Separate out on and off events.
    events_idx = events_idx.reshape((-1, 2))
    on_events = events_idx[:, 0].copy()
    off_events = events_idx[:, 1].copy()
    assert len(on_events) == len(off_events)

    # Filter out on and off transitions faster than minimum values.
    if len(on_events) > 0:
        off_duration = on_events[1:] - off_events[:-1]
        off_duration = np.insert(off_duration, 0, 1000)
        on_events = on_events[off_duration > min_off_duration]
        off_events = off_events[np.roll(off_duration, -1) > min_off_duration]

        on_duration = off_events - on_events
        on_events = on_events[on_duration >= min_on_duration]
        off_events = off_events[on_duration >= min_on_duration]
        assert len(on_events) == len(off_events)

    # Generate final status.
    status = [0] * appliance_power.size
    for on, off in zip(on_events, off_events):
        status[on: off] = [1] * (off - on)

    return status
