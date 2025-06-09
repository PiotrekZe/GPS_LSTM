import os
import numpy as np
import pandas as pd


def safe_append(arr1, arr2):
    """
    Safely append two numpy arrays along the first axis, handling empty arrays.
    """
    if arr1.shape == (0,):
        return arr2
    if arr2.size == 0:
        return arr1
    return np.append(arr1, arr2, axis=0)


def prepare_dataset(path, positions, input_size, output_size, step, ver=1):
    """
    Prepare input/output sequences for a single file, split by bridge positions.
    Args:
        path (str): Path to the CSV file.
        positions (list): [start, end] indices for bridge location.
        input_size (int): Length of input sequence.
        output_size (int): Length of output sequence.
        step (int): Step size for sliding window.
        ver (int): Version of output (1: merged, 2: separated before/after bridge).
    Returns:
        tuple: Arrays of input and output sequences.
    """
    data = pd.read_csv(path)
    # Use only columns 4 and 5 (assumed to be coordinates)
    data = np.array(data.iloc[:, [4, 5]], dtype=np.float64)
    before_bridge, after_bridge = data[: positions[0], :], data[positions[1] :, :]

    before_input, before_output = [], []
    after_input, after_output = [], []

    before_n = int((len(before_bridge) - input_size - output_size) / step)
    after_n = int((len(after_bridge) - input_size - output_size) / step)

    # Generate input/output pairs before the bridge
    for i in range(before_n):
        before_input.append(before_bridge[i * step : i * step + input_size, :])
        before_output.append(
            before_bridge[
                i * step + input_size : i * step + input_size + output_size, :
            ]
        )
    # Generate input/output pairs after the bridge
    for i in range(after_n):
        after_input.append(after_bridge[i * step : i * step + input_size, :])
        after_output.append(
            after_bridge[i * step + input_size : i * step + input_size + output_size, :]
        )

    tmp_input = safe_append(np.array(before_input), np.array(after_input))
    tmp_output = safe_append(np.array(before_output), np.array(after_output))

    if ver == 1:
        return tmp_input, tmp_output
    elif ver == 2:
        return (
            np.array(before_input),
            np.array(before_output),
            np.array(after_input),
            np.array(after_output),
        )


def read_data(input_size=100, output_size=200, step=10):
    """
    Read and aggregate GPS data from multiple rivers and segments.
    Args:
        input_size (int): Length of input sequence.
        output_size (int): Length of output sequence.
        step (int): Step size for sliding window.
    Returns:
        tuple: List of input arrays and output arrays for each river.
    """
    rivers = ["Brdowski", "Clowy", "pionierow"]
    path = "E:/AI/Mosty/"

    # Define bridge segments for each river (start, end indices)
    brdowski_segments = [
        [430, 600],
        [300, 450],
        [320, 560],
        [300, 450],
        [300, 470],
        [350, 450],
        [520, 730],
        [280, 450],
        [500, 650],
        [370, 410],
        [390, 500],
        [420, 580],
        [350, 610],
        [390, 480],
        [430, 600],
        [550, 700],
    ]
    clowy_segments = [
        [470, 680],
        [420, 600],
        [480, 730],
        [690, 950],
        [520, 710],
        [620, 920],
        [400, 600],
        [480, 650],
        [250, 410],
        [450, 590],
        [450, 600],
        [260, 470],
        [540, 800],
        [750, 850],
        [370, 500],
        [520, 790],
    ]
    pionierow_segments = [
        [490, 890],
        [250, 700],
        [690, 1050],
        [450, 850],
        [390, 1050],
        [400, 1050],
        [620, 1210],
        [400, 850],
        [600, 1140],
        [600, 950],
        [770, 1200],
        [400, 780],
        [700, 1100],
        [260, 510],
        [550, 900],
    ]
    positions = [brdowski_segments, clowy_segments, pionierow_segments]

    rivers_inputs, rivers_outputs = [], []

    for i, river in enumerate(rivers):
        tmp_path = os.path.join(path, river)
        files = os.listdir(tmp_path)
        river_input, river_output = None, None
        for j, file in enumerate(files):
            file_path = os.path.join(tmp_path, file)
            # Use segment index j//2 for each file (assumes 2 files per segment)
            input, output = prepare_dataset(
                file_path,
                positions=positions[i][j // 2],
                input_size=input_size,
                output_size=output_size,
                step=step,
                ver=1,
            )
            if input.shape == np.array([]).shape:
                continue
            if river_input is None:
                river_input = input
                river_output = output
            else:
                river_input = np.concatenate([river_input, input], axis=0)
                river_output = np.concatenate([river_output, output], axis=0)
        rivers_inputs.append(river_input)
        rivers_outputs.append(river_output)

    return rivers_inputs, rivers_outputs
