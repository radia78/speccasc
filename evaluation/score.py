import pandas as pd
import re
import argparse
import os
import glob
import evaluate

def extract_number(text):
    """
    Extracts the final numerical answer from the text.
    Prioritizes '#### <number>' (GSM8K standard)
    Then 'The answer is <number>'
    Then boxed answers.
    """
    if not isinstance(text, str):
        return None

    # 1. Standard GSM8K format: "#### 500"
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(',', '')

    # 2. "The answer is <number>" (User specified format)
    # We look for the LAST occurrence of this pattern
    matches = re.findall(r"The answer is[:\s]*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if matches:
        return matches[-1].replace(',', '')

    # 3. Boxed format: \boxed{500}
    match = re.search(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(',', '')
    
    return None

def clean_response(response):
    """
    Cleans the response by removing artifacts after the first newline.
    """
    if not isinstance(response, str):
        return ""
    if '\n' in response:
        response = response.split('\n')[0]
    return response

def score_gsm8k(df, filename):
    correct = 0
    total = 0
    
    for _, row in df.iterrows():
        response = clean_response(str(row['response']))
        ground_truth = str(row['answer'])
        
        pred = extract_number(response)
        gt = extract_number(ground_truth)
        
        is_correct = False
        if pred is not None and gt is not None:
            try:
                # Compare as floats to handle 500 vs 500.0
                if abs(float(pred) - float(gt)) < 1e-6:
                    is_correct = True
            except ValueError:
                # String comparison if float conversion fails
                if pred.strip() == gt.strip():
                    is_correct = True
        
        if is_correct:
            correct += 1
        
        total += 1
        
    accuracy = correct / total if total > 0 else 0
    print(f"File: {filename}")
    print(f"GSM8K Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

def score_cnndm(df, filename):
    print(f"Scoring CNNDM for {filename}...")
    rouge = evaluate.load('rouge')
    predictions = [clean_response(str(r)) for r in df['response']]
    references = [str(r) for r in df['answer']]
    
    results = rouge.compute(predictions=predictions, references=references)
    print(f"File: {filename}")
    print(f"Rouge-L: {results['rougeL']:.4f}")
    return results['rougeL']

def score_wmt(df, filename):
    print(f"Scoring WMT for {filename}...")
    bleu = evaluate.load('sacrebleu')
    predictions = [clean_response(str(r)) for r in df['response']]
    references = [[str(r)] for r in df['answer']] # sacrebleu expects list of lists for references
    
    results = bleu.compute(predictions=predictions, references=references)
    print(f"File: {filename}")
    print(f"BLEU: {results['score']:.4f}")
    return results['score']

def score_file(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    if 'answer' not in df.columns:
        print(f"Skipping {os.path.basename(csv_path)}: 'answer' column not found. Please re-run inference with the updated eval.py.")
        return

    filename = os.path.basename(csv_path)
    
    if 'gsm8k' in filename.lower():
        score_gsm8k(df, filename)
    elif 'cnn' in filename.lower():
        score_cnndm(df, filename)
    elif 'wmt' in filename.lower():
        score_wmt(df, filename)
    else:
        print(f"Unknown benchmark for file: {filename}. Skipping. (Filename must contain 'gsm8k', 'cnn', or 'wmt')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs='?', default="results", help="Path to CSV file or directory containing CSV files")
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, "*.csv"))
        if not files:
            print(f"No CSV results found in {args.path}")
        for file in sorted(files):
            score_file(file)
    elif os.path.isfile(args.path):
        score_file(args.path)
    else:
        print(f"Invalid path: {args.path}")
