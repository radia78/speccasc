import pandas as pd
import re
import argparse
import os
import glob
import evaluate

from prompts import (
    WMT_DE_EN_STOP_STRINGS,
    MBPP_STOP_STRINGS,
    CNN_DM_STOP_STRINGS,
    Q_A_STOP_STRINGS
)

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

    # 2. "The answer is <number>"
    # We look for the LAST occurrence of this pattern
    matches = re.findall(r"The answer is[:\s]*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if matches:
        return matches[-1].replace(',', '')

    # 3. Boxed format: \boxed{500}
    match = re.search(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(',', '')
    
    return None

def clean_qa_response(response):
    """
    Cleans the response by removing artifacts.
    The user noted that LLM outputs might contain "Q:" and extra examples.
    We want to keep the main answer but stop before the model starts hallucinating new questions.
    """
    if not isinstance(response, str):
        return ""
    
    # If the response contains "Q:", it might be starting a new example. 
    # We should cut off everything after the last valid answer and before the next Q:
    # But simply splitting by \n might be too aggressive if the model uses newlines for formatting.
    
    # Strategy:
    # 1. If we see "Q:" or "Question:", cut off there.
    if "Q:" in response:
        response = response.split("Q:")[0]
    
    return response.strip()

def clean_mbpp_response(response):
    """
    Clean strings generated from different models for MBPP program generation
    """

    # Detect for any stop strings and then remove them as necessary
    for stop_str in MBPP_STOP_STRINGS:
        if stop_str in response:
            response = response.strip(stop_str)

    # It mind start yapping like crazy, so just find the match and group and strip!
    program_cleaned_str = re.match(r'(?s).*?(?=\n[A-Z])', response)
    return program_cleaned_str.group().strip()

def score_gsm8k(df, filename):
    correct = 0
    total = 0
    
    for i, row in df.iterrows():
        response = clean_qa_response(str(row['response']))
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
        else:
            # Debug print for incorrect answers to help diagnose
            # print(f"Row {i}: Incorrect. Pred: {pred}, GT: {gt}")
            pass
        
        total += 1
        
    accuracy = correct / total if total > 0 else 0
    print(f"File: {filename}")
    print(f"GSM8K Accuracy: {accuracy:.2%} ({correct}/{total})")
    return {'accuracy': accuracy, 'correct': correct, 'total': total}

def score_cnndm(df, filename):
    print(f"Scoring CNNDM for {filename}...")
    rouge = evaluate.load('rouge')
    predictions = [clean_response(str(r)) for r in df['response']]
    references = [str(r) for r in df['answer']]
    
    results = rouge.compute(predictions=predictions, references=references)
    print(f"File: {filename}")
    print(f"Rouge-L: {results['rougeL']:.4f}")
    return results

def score_wmt(df, filename):
    print(f"Scoring WMT for {filename}...")
    bleu = evaluate.load('sacrebleu')
    predictions = [clean_response(str(r)) for r in df['response']]
    references = [[str(r)] for r in df['answer']] # sacrebleu expects list of lists for references
    
    results = bleu.compute(predictions=predictions, references=references)
    print(f"File: {filename}")
    print(f"BLEU: {results['score']:.4f}")
    return results

def run_program(program: str, test_cases: list[str]):
    # Input should be a row of the data frame
    # Run the string function immediately to see if it actually works
    try:
        exec(program)
        for t in test_cases:
            exec(t)
        return True

    except:
        return False
    
def score_mbpp(df, filename):
    correct = 0
    total = 0
    
    for _, row in df.iterrows():
        response, test_cases = row['response'], list(row['test_cases'])
        is_correct = run_program(response, test_cases)
        
        if is_correct:
            correct += 1
        else:
            # Debug print for incorrect answers to help diagnose
            # print(f"Row {i}: Incorrect. Pred: {pred}, GT: {gt}")
            pass
        
        total += 1
        
    accuracy = correct / total if total > 0 else 0
    print(f"File: {filename}")
    print(f"MBPP Accuracy: {accuracy:.2%} ({correct}/{total})")
    return {'accuracy': accuracy, 'correct': correct, 'total': total}

def score_file(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

    if 'answer' not in df.columns:
        print(f"Skipping {os.path.basename(csv_path)}: 'answer' column not found. Please re-run inference with the updated eval.py.")
        return None

    filename = os.path.basename(csv_path)
    result = None
    
    if 'gsm8k' in filename.lower():
        result = score_gsm8k(df, filename)
        result['benchmark'] = 'gsm8k'
    elif 'cnn' in filename.lower():
        result = score_cnndm(df, filename)
        result['benchmark'] = 'cnn_dm'
    elif 'wmt' in filename.lower():
        result = score_wmt(df, filename)
        result['benchmark'] = 'wmt_de_en'
    elif 'mbpp' in filename.lower():
        result = score_mbpp(df, filename)
        result['benchmark'] = 'mbpp'
    else:
        print(f"Unknown benchmark for file: {filename}. Skipping. (Filename must contain 'gsm8k', 'cnn', or 'wmt')")
        return None
        
    result['filename'] = filename
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs='?', default="results", help="Path to CSV file or directory containing CSV files")
    args = parser.parse_args()
    
    all_results = []
    
    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, "*.csv"))
        if not files:
            print(f"No CSV results found in {args.path}")
        for file in sorted(files):
            res = score_file(file)
            if res:
                all_results.append(res)
    elif os.path.isfile(args.path):
        res = score_file(args.path)
        if res:
            all_results.append(res)
    else:
        print(f"Invalid path: {args.path}")
        
    if all_results:
        # Save to scores directory
        scores_dir = "scores"
        if not os.path.exists(scores_dir):
            os.makedirs(scores_dir)
            
        output_df = pd.DataFrame(all_results)
        
        # Save summary
        output_path = os.path.join(scores_dir, "summary_scores.csv")
        
        # Reorder summary columns: benchmark, filename, then primary metrics, then rest
        summary_cols = output_df.columns.tolist()
        summary_order = ['benchmark', 'filename']
        
        # Add primary metrics if they exist
        for metric in ['accuracy', 'rougeL', 'score']:
            if metric in summary_cols:
                summary_order.append(metric)
                
        # Add remaining columns
        for c in summary_cols:
            if c not in summary_order:
                summary_order.append(c)
                
        output_df = output_df[summary_order]
        output_df.to_csv(output_path, index=False)
        print(f"\nSummary scores saved to {output_path}")
        
        # Save individual benchmark scores
        for benchmark in output_df['benchmark'].unique():
            bench_df = output_df[output_df['benchmark'] == benchmark]
            # Drop columns that are completely empty (NaN) for this benchmark
            bench_df = bench_df.dropna(axis=1, how='all')
            
            # Reorder columns: benchmark, filename, then primary metric, then rest
            cols = bench_df.columns.tolist()
            primary_metric = None
            if benchmark == 'gsm8k':
                primary_metric = 'accuracy'
            elif benchmark == 'cnn_dm':
                primary_metric = 'rougeL'
            elif benchmark == 'wmt_de_en':
                primary_metric = 'score' # BLEU score key from sacrebleu
            
            # Start with benchmark and filename
            new_order = ['benchmark', 'filename']
            if primary_metric and primary_metric in cols:
                new_order.append(primary_metric)
            
            # Add remaining columns
            for c in cols:
                if c not in new_order:
                    new_order.append(c)
            
            bench_df = bench_df[new_order]
            
            bench_path = os.path.join(scores_dir, f"{benchmark}_scores.csv")
            bench_df.to_csv(bench_path, index=False)
            print(f"Saved {benchmark} scores to {bench_path}")
