import numpy as np
import time
import librosa
import os, select
import json
from scipy.signal import savgol_filter
import ruptures as rpt
from collections import Counter
from dtaidistance import dtw



ref_curve = np.load("ref_curve.npy")

def normalize(x):
    std = np.std(x)
    if std == 0:
        return np.zeros_like(x)   # or return x
    return (x - np.mean(x)) / std
def derivative(x):
    return np.gradient(x)
    

def find_silent_segment(signal, threshold=1e-6, min_length=1000):
    signal = np.asarray(signal)

    best_start = 0
    best_len = 0

    current_start = None

    for i, x in enumerate(signal):
        if abs(x) < threshold:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                length = i - current_start
                if length > best_len and length >= min_length:
                    best_len = length
                    best_start = current_start
                current_start = None

    # handle case where signal ends in silence
    if current_start is not None:
        length = len(signal) - current_start
        if length > best_len and length >= min_length:
            best_len = length
            best_start = current_start

    return best_start, best_start + best_len, best_len

    
EXTRACT_DIR = "extract"

sr = 44100
n_fft = 512*1
hop_length = 128*1
window = 'hann'

def process_file(path, ref_curve):
    try:
        print(f"Processing file: {path}")
        audio, sr = librosa.load(path, sr=None)

        # ----------------------------------------------------------
      
        # Complex STFT
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
        S_magnitude = np.abs(S)
        S_db = librosa.amplitude_to_db(S_magnitude, ref=np.max)
        frequencies = librosa.fft_frequencies(sr=sr)
        E = np.abs(S[:, 1:] - S[:, :-1])**2
        E_db = 20 * np.log10(E + 1e-12)

        # CLIPPING CHECK
        threshold = 0.999
        clipped_samples = np.where(np.abs(audio) >= threshold)[0]
        clipped_frames = np.unique(clipped_samples // hop_length)

        jump = np.abs(np.diff(audio))

        med = np.median(jump)
        mad = np.median(np.abs(jump - med))
        
        threshold = med + 300 * mad
        
        anomalies = np.where(jump > threshold)[0]
        jump = []
        for i in anomalies:
            jump.append(int(i/hop_length))

            
    
        A = E_db.shape[1]     
        nums = np.sort(np.random.choice(A, 1000, replace=False))
        E_smooth = savgol_filter(E_db, window_length=11, polyorder=3, axis=0)
        
        def process_column(i):
            y_smooth = E_smooth[:, i]
        
            algo = rpt.Pelt(model="rbf").fit(y_smooth)
            cp = np.asarray(algo.predict(pen=10))
        
            cp = np.where(cp == 257, 256, cp)
        
            vals = E_db[cp, i]
            mask = (vals > -200) & (vals < -140)
        
            return cp[mask]
        
        results = Parallel(n_jobs=-1)(
            delayed(process_column)(i) for i in nums
        )
        
        new_change_point = np.concatenate(results).tolist()
        
        counter = Counter(new_change_point)
        
        B = 2
        
        while True:
            top_numbers = [num for num, count in counter.most_common(B)]
            top_numbers.sort()
        
            F1 = top_numbers[0]
            E1 = top_numbers[-1]
        
            if E1 - F1 >= 20 or B >= len(counter):
                break
        
        if E1 - F1 >= 150:
            F = F1
            E = F + 40
            
        else:
            F = F1
            E = E1

            
        curve1_n = normalize(ref_curve)
        win_len = len(curve1_n)
        indx = []
        ind = []
        INX = []
   
        # collect valid curves (same condition as original)
        for j in range(A):
            curve2_n = normalize(E_db[: , j][F:E])
            if (S_db[F:E, j] > -80).mean() >= 0.1: 
               
                distances = []        
                for i in range(len(curve2_n) - win_len):
                    segment = curve2_n[i:i+win_len]
                    d = dtw.distance(derivative(curve1_n), derivative(segment))
                    distances.append(d)
                
                distances = np.array(distances)
                best_distance = np.min(distances)
                if best_distance < 1.2:
                    ind.append(j)
           
        for i in ind:
            if any(i + d in clipped_frames for d in range(-3, 4)):
                continue
            indx.append(i)
        
        indx = [i for i in indx if i not in (0, 1, A - 1)]
        offsets = [o for o in range(-6, 7) if o != 0]

        for i in indx:
            if all(np.std(E_db[F:E, i + o]) != 0 for o in offsets):
                INX.append(i)
   
        start, end, length = find_silent_segment(audio)
        smp_hld = []
        j = 0
        THR = 0.0001
        if start > 0:
            if abs(audio[start-1]-audio[start]) > THR:
                j = j+1
                smp_hld.append(j)
            
        if end != 0 and end < len(audio):
            if abs(audio[end-1]- audio[end]) > THR:
                j = j+1
                smp_hld.append(j)
 
        jump_cut = []

        for i in INX:
            if i in jump:
                jump_cut.append(i)
        
        error_type = None
        first_frame = None
        
        # 1️⃣ HIGHEST priority: Cut Segment
        if INX:
            if jump_cut:
                error_type = "Jump_discontiniuty"
            else:
                error_type = "Cut Segment"
                first_frame = INX[0]
     
        elif smp_hld:
            error_type = "Hold_sample_error"

        else:
            error_type = "Intact"

        return error_type, first_frame

    except Exception as e:
        print(f"❗ Error reading {path}: {e}")
        return None

def is_file_complete(path, stable_time=1.0, check_interval=0.2):
    """
    Returns True if the file size is stable for `stable_time` seconds.
    """
    if not os.path.exists(path):
        return False
    
    elapsed = 0
    prev_size = os.path.getsize(path)
    
    while elapsed < stable_time:
        time.sleep(check_interval)
        elapsed += check_interval
        curr_size = os.path.getsize(path)
        if curr_size != prev_size:
            # File is still growing
            prev_size = curr_size
            elapsed = 0  # reset timer
    return True

def wait_for_file_complete(path, stable_time=1.0, check_interval=0.2, timeout=60):
    """
    Wait until a file is fully written and unlocked.
    Returns True if ready, False if timeout.
    """
    start_time = time.time()
    last_size = -1
    stable_elapsed = 0

    while True:
        if not os.path.exists(path):
            return False

        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1

        if size == last_size:
            stable_elapsed += check_interval
        else:
            stable_elapsed = 0
            last_size = size

        if stable_elapsed >= stable_time:
            try:
                with open(path, 'rb'):
                    return True
            except (OSError, PermissionError):
                stable_elapsed = 0  # reset if file is locked

        if (time.time() - start_time) > timeout:
            return False

        time.sleep(check_interval)

def safe_delete(path, retries=10, delay=0.5):
    for attempt in range(retries):
        try:
            os.remove(path)
            return True
        except PermissionError:
            time.sleep(delay)
    return False

def main(ref_curve):
    print("=== SCRIPT STARTED ===")
    print("Python working directory:", os.getcwd())
    ERROR_DIR = os.path.join(EXTRACT_DIR, "error")
    os.makedirs(ERROR_DIR, exist_ok=True)
    INFO_JSON = os.path.join(EXTRACT_DIR, "information.json")

    while True:
        try:
            files = sorted(f for f in os.listdir(EXTRACT_DIR) if f.lower().endswith(".wav"))

            for fname in files:
                full = os.path.join(EXTRACT_DIR, fname)

                if not wait_for_file_complete(full, stable_time=1.0, timeout=60):
                    print(f"File {fname} is still being written. Waiting...")
                    continue

                #result = process_file(full, model)
                error_type, first_frame = process_file(full, ref_curve)
                time_position = None
                if first_frame is not None:
                    time_position = (first_frame * hop_length) / sr
                    
                feature_dict = {                    
                    "filename": full,
                    "Error": error_type,
                    "Time_Position": time_position                   
                }

                #print(f"Process result for {fname}: {error_type}")
                print(f"Process result: {error_type}")

                if error_type == "Intact":
                    if safe_delete(full):
                        print(f"Deleted file: {fname}")
                    else:
                        print(f"❗ Could not delete {fname} after multiple attempts. Will retry later.")
                        
                else:                    
                    # ---- JSON logging ----
                    if os.path.exists(INFO_JSON):
                        try:
                            with open(INFO_JSON, "r") as f:
                                data = json.load(f)
                                if not isinstance(data, list):
                                    data = []
                        except json.JSONDecodeError:
                            data = []
                    else:
                        data = []
                
                    data.append(feature_dict)
                
                    with open(INFO_JSON, "w") as f:
                        json.dump(data, f, indent=4)
                
                    safe_error = error_type.replace(" ", "_")
                    base_name = f"{os.path.splitext(fname)[0]}_{safe_error}"
                    new_path = os.path.join(ERROR_DIR, base_name + ".wav")
                    
                    counter = 1
                    while os.path.exists(new_path):
                        new_path = os.path.join(ERROR_DIR, f"{base_name}_{counter}.wav")
                        counter += 1

               
                    try:
                        os.rename(full, new_path)
                        print(f"[ERROR] File moved to: {new_path}")
                    except Exception as e:
                        print(f"❗ Could not move {fname}: {e}")


                #PROCESSED.add(full)

            time.sleep(1)

        except Exception as e:
            print("❗ LOOP ERROR:", e)
            time.sleep(1)

            
if __name__ == "__main__":
    
    main(ref_curve)
