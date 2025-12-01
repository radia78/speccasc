import pandas as pd
import re
import argparse
import os
import glob

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
    
    # 4. Fallback: Look for the very last number in the text
    # This is risky but often necessary for weak models that just output the number
    # We'll try to be a bit conservative and only take it if it's at the end of the string
    # match = re.search(r"(-?[\d,]+(?:\.\d+)?)\s*$", text)
    # if match:
    #     return match.group(1).replace(',', '')

    return None

def score_gsm8k(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    if 'answer' not in df.columns:
        print(f"Skipping {os.path.basename(csv_path)}: 'answer' column not found. Please re-run inference with the updated eval.py.")
        return

    correct = 0
    total = 0
    
    results = []

    for _, row in df.iterrows():
        response = str(row['response'])
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
        results.append({
            'response': response,
            'answer': ground_truth,
            'pred_extracted': pred,
            'gt_extracted': gt,
            'is_correct': is_correct
        })
        
    accuracy = correct / total if total > 0 else 0
    print(f"File: {os.path.basename(csv_path)}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Optionally save detailed scoring results
    # output_path = csv_path.replace('.csv', '_scored.csv')
    # pd.DataFrame(results).to_csv(output_path, index=False)
    
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs='?', default="results", help="Path to CSV file or directory containing CSV files")
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, "*gsm8k*.csv"))
        if not files:
            print(f"No GSM8K results found in {args.path}")
        for file in sorted(files):
            score_gsm8k(file)
    elif os.path.isfile(args.path):
        score_gsm8k(args.path)
    else:
        print(f"Invalid path: {args.path}")
