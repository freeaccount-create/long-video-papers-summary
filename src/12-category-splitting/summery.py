import os
import re
import ast
import unicodedata
import numpy as np
from collections import defaultdict

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
NBSP_SET = "\u00A0\u202F\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A"
WSJUNK = r"[\s" + NBSP_SET + r"\u200B-\u200D\u2060\uFEFF\x00-\x1F\x7F]*"

def _load_out_text(path: str) -> str:
    b = open(path, "rb").read()
    b = b.replace(b"\x00", b"").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    s = b.decode("utf-8", errors="replace")
    s = ANSI_RE.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = ZERO_WIDTH_RE.sub("", s)
    for ch in NBSP_SET:
        s = s.replace(ch, " ")
    return s

def _loose_key_pattern(key: str) -> str:
    return "".join(re.escape(c) + WSJUNK for c in key)

def _extract_brace_block(text: str, key: str) -> str | None:
    pat = rf"{_loose_key_pattern(key)}[:：]{WSJUNK}(\{{)"
    m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    start = m.start(1)
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def extract_generality_data_from_out_file(file_path):
    try:
        content = _load_out_text(file_path)
    except Exception as e:
        print(f"[WARN] read fail: {file_path} -> {e}")
        return None

    block = _extract_brace_block(content, "Generality_per_class")
    if not block:
        print(f"[WARN] can not find Generality_per_class: {file_path}")
        return None
    try:
        generality_dict = ast.literal_eval(block)
        if not isinstance(generality_dict, dict):
            print(f"[WARN] not dict: {file_path}")
            return None
        return generality_dict
    except Exception as e:
        print(f"[WARN] fail {file_path}: {e}")
        return None

def extract_locality_from_file(file_path):
    pattern = re.compile(rf"Locality: \s*([0-9]*\.?[0-9]+)")
    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(",")]
                return values
    return None

def extract_job_from_file(file_path):
    pattern = re.compile(r"Split target:\s*(.+)")
    with open(file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return match.group(1).strip()
    return None

def compute_total_accuracy_from_folder(folder_path):
    generality_avgs = {} # files：generality_avg
    generality_all_tp = defaultdict(list) # [[],[],[],[]]
    generality_all_total = defaultdict(list)
    locality_avgs = {}
    localitys_all = defaultdict(list)
    for root, dirs, files in os.walk(folder_path):
        generality_tp, generality_total = [], []
        localitys = []
        seeds = []
        for filename in files:
            if not filename.endswith(".out"):
                continue
            file_path = os.path.join(folder_path, os.path.basename(root), filename)
            sub_folder = os.path.basename(root)
            # get generality dict
            generality_dict = extract_generality_data_from_out_file(file_path)
            if not generality_dict:
                print(sub_folder, "wrong")
                exit()
            temp_tp = []
            temp_total = []
            for value in generality_dict.values():
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    temp_tp.append(int(value[1]))
                    temp_total.append(int(value[2]))
                else:
                    print(f"[WARN] format wrong {filename}: {value!r}")
            # here is the data for one file (one seed for one coarse class)
            generality_tp.append(sum(temp_tp))
            generality_total.append(sum(temp_total))
            l = extract_locality_from_file(file_path)
            if l == None:
                print(sub_folder, "wrong")
                exit()
            else:
                l=l[-1]
            localitys.append(l)
            seeds.append(extract_job_from_file(file_path).split("_")[-1])
        
        # now is the result for one subfolder which is the result from different seed for one coarse class
        if generality_tp and generality_total and localitys:
            # for each split target get accuracy
            acc = np.array(generality_tp) / np.array(generality_total)
            avg = np.mean(acc)
            std = np.std(acc, ddof=1)
            generality_avgs[sub_folder] = (round(float(avg), 3), round(float(std), 3))
            avg = np.mean(np.array(localitys))
            std = np.std(np.array(localitys), ddof=1)
            locality_avgs[sub_folder] = (round(float(avg), 3), round(float(std), 3))
            # for all seed and all split target
            for i, seed in enumerate(seeds):
                generality_all_tp[seed].append(generality_tp[i])
                generality_all_total[seed].append(generality_total[i])
                localitys_all[seed].append(localitys[i])
    
    temp_generalitys_total = []
    temp_localitys_total = []
    for key, value in generality_all_tp.items():
        generality_total_tp = sum(value)
        generality_total_total = sum(generality_all_total[key])
        temp_generalitys_total.append(generality_total_tp / generality_total_total)
        temp_localitys_total.append(sum(localitys_all[key])/len(localitys_all[key]))
    # total accuracy
    avg = np.mean(np.array(temp_generalitys_total))
    std = np.std(np.array(temp_generalitys_total), ddof=1)
    generality_avgs['All'] = (round(float(avg), 3), round(float(std), 3))
    avg = np.mean(np.array(temp_localitys_total))
    std = np.std(np.array(temp_localitys_total), ddof=1)
    locality_avgs['All'] = (round(float(avg), 3), round(float(std), 3))

    # caculate the mean of two metric
    mean_total = (np.array(temp_generalitys_total) + np.array(temp_localitys_total)) / 2
    mean_avg = np.mean(mean_total)
    mean_std = np.std(mean_total, ddof=1)
    return generality_avgs, locality_avgs, (mean_avg, mean_std)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute generality_total accuracy from Slurm .out files.")
    parser.add_argument("folder", type=str, help="Path to folder containing all .out files")

    args = parser.parse_args()
    generality_avgs, locality_avgs, (mean_avg, mean_std) = compute_total_accuracy_from_folder(args.folder)
    print(generality_avgs)
    print(locality_avgs)
    print(mean_avg, mean_std)

    output_file = os.path.join(args.folder, "results.txt")
    with open(output_file, "w") as f:
        f.write("Generality Results:\n")
        for key, (avg, std) in generality_avgs.items():
            f.write(f"{key}\tavg={avg:.3f}\tstd={std:.3f}\n")

        f.write("\nLocality Results:\n")
        for key, (avg, std) in locality_avgs.items():
            f.write(f"{key}\tavg={avg:.3f}\tstd={std:.3f}\n")

        f.write("\nMean Results:\n")
        f.write(f"\tmean_avg={mean_avg:.3f}\tmean_std={mean_std:.3f}\n")

    print(f"result save in: {output_file}")
