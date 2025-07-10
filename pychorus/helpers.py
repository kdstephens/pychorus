import librosa
import numpy as np
import scipy.signal
import soundfile as sf


from pychorus.similarity_matrix import TimeTimeSimilarityMatrix, TimeLagSimilarityMatrix, Line
from pychorus.constants import N_FFT, SMOOTHING_SIZE_SEC, LINE_THRESHOLD, MIN_LINES, \
    NUM_ITERATIONS, OVERLAP_PERCENT_MARGIN


def local_maxima_rows(denoised_time_lag):
    """Find rows whose normalized sum is a local maxima"""
    row_sums = np.sum(denoised_time_lag, axis=1)
    divisor = np.arange(row_sums.shape[0], 0, -1)
    normalized_rows = row_sums / divisor
    local_minima_rows = scipy.signal.argrelextrema(normalized_rows, np.greater)
    return local_minima_rows[0]


def detect_lines(denoised_time_lag, rows, min_length_samples):
    """Detect lines in the time lag matrix. Reduce the threshold until we find enough lines"""
    cur_threshold = LINE_THRESHOLD
    for _ in range(NUM_ITERATIONS):
        line_segments = detect_lines_helper(denoised_time_lag, rows,
                                            cur_threshold, min_length_samples)
        if len(line_segments) >= MIN_LINES:
            return line_segments
        cur_threshold *= 0.95

    return line_segments


def detect_lines_helper(denoised_time_lag, rows, threshold,
                        min_length_samples):
    """Detect lines where at least min_length_samples are above threshold"""
    num_samples = denoised_time_lag.shape[0]
    line_segments = []
    cur_segment_start = None
    for row in rows:
        if row < min_length_samples:
            continue
        for col in range(row, num_samples):
            if denoised_time_lag[row, col] > threshold:
                if cur_segment_start is None:
                    cur_segment_start = col
            else:
                if (cur_segment_start is not None
                   ) and (col - cur_segment_start) > min_length_samples:
                    line_segments.append(Line(cur_segment_start, col, row))
                cur_segment_start = None
    return line_segments


def count_overlapping_lines(lines, margin, min_length_samples):
    """Look at all pairs of lines and see which ones overlap vertically and diagonally"""
    line_scores = {line: 0 for line in lines}
    # Iterate over all pairs of lines
    for line_1 in lines:
        for line_2 in lines:
            # If line_2 completely covers line_1 (with some margin), line_1 gets a point
            lines_overlap_vertically = (
                line_2.start < (line_1.start + margin)) and (
                    line_2.end > (line_1.end - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)
            lines_overlap_diagonally = (
                (line_2.start - line_2.lag) < (line_1.start - line_1.lag + margin)) and (
                    (line_2.end - line_2.lag) > (line_1.end - line_1.lag - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)
            if lines_overlap_vertically or lines_overlap_diagonally:
                line_scores[line_1] += 1
    return line_scores


def best_segments(line_scores, top_n=1):
    """Return the best lines, sorted first by chorus matches, then by duration"""
    lines_to_sort = [(line, score, line.end - line.start) for line, score in line_scores.items()]
    lines_to_sort.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [line for line, _, _ in lines_to_sort[:top_n]]


def draw_lines(num_samples, sample_rate, lines):
    """Debugging function to draw detected lines in black"""
    lines_matrix = np.zeros((num_samples, num_samples))
    for line in lines:
        lines_matrix[line.lag:line.lag + 4, line.start:line.end + 1] = 1

    # Import here since this function is only for debugging
    import librosa.display
    import matplotlib.pyplot as plt
    librosa.display.specshow(
        lines_matrix,
        y_axis='time',
        x_axis='time',
        sr=sample_rate / (N_FFT / 2048))
    plt.colorbar()
    plt.set_cmap("hot_r")
    plt.show()


def create_chroma(input_file, n_fft=N_FFT):
    """
    Generate the notes present in a song

    Returns: tuple of 12 x n chroma, song wav data, sample rate (usually 22050)
             and the song length in seconds
    """
    y, sr = librosa.load(input_file)
    song_length_sec = y.shape[0] / float(sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft))**2
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    return chroma, y, sr, song_length_sec


def find_chorus(chroma, sr, song_length_sec, clip_length, top_n=1):
    """
    Find the most repeated chorus

    Args:
        chroma: 12 x n frequency chromogram
        sr: sample rate of the song, usually 22050
        song_length_sec: length in seconds of the song (lost in processing chroma)
        clip_length: minimum length in seconds we want our chorus to be (at least 10-15s)
        top_n: Number of choruses we want to return

    Returns: Time in seconds of the start of the best choruses
    """
    num_samples = chroma.shape[1]

    time_time_similarity = TimeTimeSimilarityMatrix(chroma, sr)
    time_lag_similarity = TimeLagSimilarityMatrix(chroma, sr)

    # Denoise the time lag matrix
    chroma_sr = num_samples / song_length_sec
    smoothing_size_samples = int(SMOOTHING_SIZE_SEC * chroma_sr)
    time_lag_similarity.denoise(time_time_similarity.matrix, smoothing_size_samples)

    # Detect lines in the image
    clip_length_samples = clip_length * chroma_sr
    candidate_rows = local_maxima_rows(time_lag_similarity.matrix)
    lines = detect_lines(time_lag_similarity.matrix, candidate_rows, clip_length_samples)

    if len(lines) == 0:
        print("No choruses were detected. Try a smaller search duration")
        return []
    line_scores = count_overlapping_lines(lines, OVERLAP_PERCENT_MARGIN * clip_length_samples, clip_length_samples)
    top_segments = best_segments(line_scores, top_n=top_n)
    
    return [line.start / chroma_sr for line in top_segments]


def find_and_output_chorus(input_file, output_file, clip_length=15, top_n=1):
    """
    Finds the most repeated chorus from input_file and outputs to output file.

    Args:
        input_file: string specifying the input file
        output_file: string where to write the chorus (wav only)
            None means don't write anything
        clip_length: minimum length in seconds of the chorus
        top_n: Number of choruses to return

    Returns: Time in seconds of the start of the best choruses as a list (e.g.: [45.21, 89.37, 132.52])
    """
    chroma, song_wav_data, sr, song_length_sec = create_chroma(input_file)
    chorus_starts = find_chorus(chroma, sr, song_length_sec, clip_length, top_n=top_n)
    if not chorus_starts:
        return []
    
    for i, chorus_start in enumerate(chorus_starts):
        print("Chorus {} found at {:0.0f} min {:.2f} sec".format(
            i+1, chorus_start // 60, chorus_start % 60))

        if output_file is not None and i == 0:
            chorus_wave_data = song_wav_data[int(chorus_start*sr) : int((chorus_start+clip_length)*sr)]
            sf.write(output_file, chorus_wave_data, sr)

    return chorus_starts
