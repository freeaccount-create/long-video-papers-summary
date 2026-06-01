import os
import json
import random
import ast
import re
import subprocess
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from tqdm import tqdm
import torch.multiprocessing as mp

try:
    from video_bench.models.qwen2vl import Qwen2VL 
except ImportError:
    print("failed to load qwen2vl")

try:
    from video_bench.models.mammoth_vl import MAmmoTH_VL 
except ImportError:
    print("failed to load mammoth_vl")

try:
    from video_bench.models.gptapi import Gpt 
except ImportError:
    print("failed to load gpt")

try:
    from video_bench.models.gemini import Gemini
except ImportError:
    print("failed to load gemini")

try:
    from video_bench.models.llava_video_image import LLaVA_Video
except ImportError:
    print("failed to load llava_video_image")

try:
    from video_bench.models.llamavid_7b import LLamaVID 
except ImportError:
    print("failed to load llamavid")

try:
    from video_bench.models.llavamini_8b import LLaVAMini 
except ImportError:
    print("failed to load llavamini")

try:
    from video_bench.models.longva_7b import LongVA 
except ImportError:
    print("failed to load longva")

try:
    from video_bench.models.longvila import LongVILA 
except ImportError:
    print("failed to load longvila")

try:
    from video_bench.models.nvila import NVILA 
except ImportError:
    print("failed to load nvila")

try:
    from video_bench.models.longvu_7b import LongVU 
except ImportError:
    print("failed to load longvu")

try:
    from video_bench.models.mplug_owl3 import mPluGOWL3 
except ImportError:
    print("failed to load mplug_owl3")

try:
    from video_bench.models.internvl2_5 import Internvl2_5
except ImportError:
    print("failed to load internvl2_5")

try:
    from video_bench.models.internvl2_5_lmdeploy import Internvl2_5
except ImportError:
    print("failed to load internvl2_5_lmdeploy")

try:
    from video_bench.models.internvl2 import Internvl2
except ImportError:
    print("failed to load internvl2")

try:
    from video_bench.models.llava_ov import LLaVA
except ImportError:
    print("failed to load llava")
    
try:
    from video_bench.models.llava_video import LLaVA_Video
except ImportError:
    print("failed to load llava_video")

try:
    from video_bench.models.phi3_5 import Phi3_5
except ImportError:
    print("failed to load phi3_5")

try:
    from video_bench.models.phi4 import Phi4
except ImportError:
    print("failed to load phi4")

try:
    from video_bench.models.llava_next_video import LLaVA_NV
except ImportError:
    print("failed to load LLaVA_NV")

try:
    from video_bench.models.minicpmv import MiniCPMV
except ImportError:
    print("failed to load minicpmv")

try:
    from video_bench.models.minicpmo import MiniCPMO
except ImportError:
    print("failed to load minicpmo")

try:
    from video_bench.models.aria import Aria
except ImportError:
    print("failed to load aria")

try:
    from video_bench.models.qwen2_5vl import Qwen2_5VL
except ImportError:
    print("failed to load qwen2_5vl")

try:
    from video_bench.models.videollama3 import VideoLlama3
except ImportError:
    print("failed to load videollama3")
    
try:
    from video_bench.models.internvideo2_5 import InternVideo2_5
except ImportError:
    print("failed to load internvideo2_5")

try:
    from video_bench.models.ola_7b import OLA
except ImportError:
    print("failed to load ola")

# 常量定义
MULTI_CHOICE_PROMPT = "Answer with the option's letter from the given choices directly."
PROMPT = (
    "We provide you with a video that has been divided into {frame_num} evenly spaced frames spanning {duration} seconds. Please answer the question based solely on the content of these frames."
)

# === 工具函数 ===

def format_question(question, options):
    formatted_options = "\n".join([f"{key}. {value}" for key, value in options.items()])
    return f"{question}？\n{formatted_options}"

def convert_to_multiple_choice(data):
    def shuffle_options(correct, distractors):
        options = distractors + [correct]
        random.shuffle(options)
        return options

    def get_option_letter(index):
        return chr(ord('a') + index)

    multiple_choice_questions = []

    for item in data:
        correct_answer = item['answer']
        distractors = [d for d in item['distractors'] if d.strip()]
        all_options = shuffle_options(correct_answer, distractors)
        correct_index = all_options.index(correct_answer)
        correct_letter = get_option_letter(correct_index)
        options = {get_option_letter(i): opt for i, opt in enumerate(all_options)}
        text = format_question(item["question"], options) + "\n" + MULTI_CHOICE_PROMPT

        question_item = {
            "question": item["question"],
            "data_id": item["data_id"],
            "question_type": item["question_type"],
            "granularity": item["granularity"],
            "options": options,
            "text": text,
            "correct_option": correct_letter,
            "answer": correct_answer,
        }
        multiple_choice_questions.append(question_item)

    return multiple_choice_questions

def load_questions_from_jsonl(question_file):

    all_data = {}
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for video_key, questions_list in obj.items():
                processed_questions = convert_to_multiple_choice(questions_list)
                all_data.setdefault(video_key, []).extend(processed_questions)
    return all_data

def get_video_duration(video_path):
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)
        return float(metadata['format']['duration'])
    except Exception as e:
        print(f"Failed to get video duration for {video_path}: {e}")
        return None

def is_text_similar(text1, text2, threshold=0.7):
    similarity = SequenceMatcher(None, text1, text2).ratio()
    return similarity > threshold

def check_answer(response, answer):

    all_choices = [chr(i) for i in range(ord('a'), ord('j') + 1)]
    if response is None:
        return False, random.choice(all_choices)

    response = response.strip(".,!?;:'").lower()
    response = f" {response} "

    match = re.search(r'<answer>:\s*(\w)', response)
    if match:
        extracted = match.group(1)
        return (extracted == answer), extracted

    candidates = []
    for choice in all_choices:
        if f"({choice})" in response or f" {choice} " in response or f"{choice}." in response:
            candidates.append(choice)
    pred = candidates[-1] if candidates else random.choice(all_choices)
    return (pred == answer), pred

def load_existing_results(output_file):

    processed_ids = set()
    total_answers = valid_answers = correct_answers = 0
    question_type_stats = defaultdict(lambda: {"total": 0, "valid": 0, "correct": 0, "correct_rate": 0.0})
    question_type_stats["total"] = {"total": 0, "valid": 0, "correct": 0, "correct_rate": 0.0}

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                result = json.loads(line.strip())
                for video_key, results in result.items():
                    for item in results:
                        processed_ids.add((video_key, item["data_id"]))
                        if item.get("model_response") != "null":
                            valid_answers += 1
                            question_type_stats[item["question_type"]]["valid"] += 1
                            question_type_stats["total"]["valid"] += 1
                        total_answers += 1
                        question_type_stats[item["question_type"]]["total"] += 1
                        question_type_stats["total"]["total"] += 1
                        if item.get("is_true"):
                            correct_answers += 1
                            question_type_stats[item["question_type"]]["correct"] += 1
                            question_type_stats["total"]["correct"] += 1

    for stats in question_type_stats.values():
        if stats["total"] > 0:
            stats["correct_rate"] = stats["correct"] / stats["total"]
    return processed_ids, total_answers, valid_answers, correct_answers, question_type_stats

def initialize_model(args):
    model_name = args.model_name
    model_path = args.model_path

    kwargs = {"model_path": model_path}
    if "vllm" in model_name:
        from video_bench.registry import get_model_vllm
        ModelClass = get_model_vllm(model_name)
    else:
        from video_bench.registry import get_model
        ModelClass = get_model(model_name)
    model = ModelClass(**kwargs)

    if args.nframes:
        model.set_frame_num(int(args.nframes))
    return model

# === 单个视频处理 ===
def process_video_questions(
    video_key,
    questions,
    model,
    video_dir,
    args,
    processed_ids,
    stats,
    target_resolution,
    keep_aspect_ratio,
    min_pixels,
    max_pixels
):
    results = {video_key: []}
    video_path = os.path.join(video_dir, f"{video_key}.mp4")
    if not os.path.exists(video_path):
        print(f"file is not exist: {video_path}")
        return results
    video_duration = get_video_duration(video_path)
    for q in questions:
        if (video_key, q["data_id"]) in processed_ids:
            print(f"processed: {video_key}, {q['data_id']}")
            continue

        stats["total"]["total"] += 1
        stats[q["question_type"]]["total"] += 1

        if isinstance(target_resolution, str):
            target_resolution = target_resolution.replace(" ", "")

        if isinstance(target_resolution, str):
            target_resolution = ast.literal_eval(target_resolution)
        text = q["text"]
        text = PROMPT.format(duration=video_duration, frame_num=int(args.nframes)) + text

        model_output = None
        try:                
            if target_resolution:
                model_output = model.generate_video_only_res(video_path, text, target_resolution)
            else:
                model_output = model.generate_video_only(video_path, text, args.nframes)
        except Exception as e:
            print(f"error: {e}")

        if model_output:
            stats["total"]["valid"] += 1
            stats[q["question_type"]]["valid"] += 1

        is_true, model_answer = check_answer(model_output, q["correct_option"])
        if model_output and is_true:
            stats["total"]["correct"] += 1
            stats[q["question_type"]]["correct"] += 1

        for label, choice in q["options"].items():
            print(f"  {label}. {choice}")
        correct_str = f"{q['correct_option']}. {q['options'][q['correct_option']]}"

        result = {
            "data_id": q["data_id"],
            "question": q["question"],
            "question_type": q["question_type"],
            "granularity": q["granularity"],
            "choices": q["options"],
            "model_answer": model_answer,
            "correct_option": correct_str,
            "is_true": is_true,
            "model_response": model_output,
        }
        results[video_key].append(result)
    return results

def save_results(output_file, results):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")

def update_stats_file(output_file, question_type_stats):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(question_type_stats, ensure_ascii=False) + "\n")

# === 主函数 ===
def main():
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Process parameters and run video QA.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory of videos")
    parser.add_argument("--question_file", type=str, required=True, help="Path to the JSONL question file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--nframes", type=str, required=True, help="Number of frames")
    parser.add_argument("--target_resolution", type=lambda x: ast.literal_eval(x), default=None, help="Target resolution, e.g. '(640,480)'")
    parser.add_argument("--keep_aspect_ratio", type=str, default="true", help="Whether to keep aspect ratio (true/false)")
    parser.add_argument("--min_pixels", type=str, default=None, help="Minimum pixels, e.g. '640 * 480'")
    parser.add_argument("--max_pixels", type=str, default=None, help="Maximum pixels, e.g. '1920 * 1080'")
    args = parser.parse_args()

    args.keep_aspect_ratio = args.keep_aspect_ratio.lower() in ["true", "1", "yes"]

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    file_base = os.path.splitext(os.path.basename(args.question_file))[0]
    output_base = os.path.splitext(os.path.basename(args.output_file))[0]
    output_file = os.path.join(output_dir, f"{file_base}_{output_base}.jsonl")

    nframes = args.nframes
    target_resolution = args.target_resolution
    keep_aspect_ratio = args.keep_aspect_ratio

    def calc_pixels(pixels_str):
        if pixels_str:
            if isinstance(pixels_str, str) and ' * ' in pixels_str:
                parts = pixels_str.split(' * ')
                return int(parts[0]) * int(parts[1])
            return int(pixels_str)
        return None

    min_pixels = calc_pixels(args.min_pixels)
    max_pixels = calc_pixels(args.max_pixels)

    question_data = load_questions_from_jsonl(args.question_file)
    processed_ids, total_answers, valid_answers, correct_answers, question_type_stats = load_existing_results(output_file)

    model = initialize_model(args)
    if nframes:
        print(f"Using {nframes} frames")
    print(f"Output file: {output_file}")

    for video_key, questions in tqdm(question_data.items(), total=len(question_data), desc="Processing Videos"):
        video_results = process_video_questions(
            video_key, questions, model, args.video_dir, args,
            processed_ids, question_type_stats,
            target_resolution, keep_aspect_ratio, min_pixels, max_pixels
        )
        if video_results.get(video_key):
            save_results(output_file, video_results)

    for stats in question_type_stats.values():
        if stats["total"] > 0:
            stats["correct_rate"] = stats["correct"] / stats["total"]
    update_stats_file(output_file, question_type_stats)


if __name__ == "__main__":
    main()
