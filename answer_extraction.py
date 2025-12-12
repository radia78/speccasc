import re

# This code was 
def extract_gsm8k_answer(text: str) -> str | None:
    """
    Extracts the numerical answer from a GSM8K completion string.
    
    This function prioritizes the standard '####' delimiter used in GSM8K 
    few-shot prompts. If found, it cleans the subsequent text to return 
    only the numeric value (handling commas, currency symbols, etc.).

    Args:
        text (str): The full output string from the LLM.

    Returns:
        str | None: The extracted number as a string (e.g., "42", "10.5"), 
                    or None if no valid answer is found.
    """
    
    # Pattern to match numbers: optional negative sign, digits, optional commas, optional decimal
    # We essentially look for things that look like numbers (e.g., -1,200.50)
    number_pattern = r'-?[\d,]+(?:\.\d+)?'

    # ------------------------------------------------------------------
    # Strategy 1: strict GSM8K format (look for '####')
    # ------------------------------------------------------------------
    if "####" in text:
        # Split by the delimiter and take the part after it
        candidate = text.split("####")[-1].strip()
        
        # Remove common non-numeric formatting often found after ####
        # e.g., "#### $50" -> "50", "#### 50%" -> "50"
        # We extract the first valid number sequence found after the delimiter.
        match = re.search(number_pattern, candidate)
        if match:
            # Remove commas (e.g., "1,200" -> "1200") for pure numerical value
            return match.group().replace(',', '')

    # ------------------------------------------------------------------
    # Strategy 2: LaTeX Boxed format (common in math fine-tunes)
    # ------------------------------------------------------------------
    # Looks for \boxed{answer}
    boxed_match = re.search(r'\\boxed\s*\{([^}]+)\}', text)
    if boxed_match:
        candidate = boxed_match.group(1).strip()
        match = re.search(number_pattern, candidate)
        if match:
            return match.group().replace(',', '')

    # ------------------------------------------------------------------
    # Strategy 3: Heuristic Fallback (Last number in text)
    # ------------------------------------------------------------------
    # If standard delimiters fail, many eval scripts fallback to taking 
    # the very last number found in the text.
    
    # Find all numbers in the text
    all_numbers = re.findall(number_pattern, text)
    if all_numbers:
        # Return the last one found, cleaned of commas
        return all_numbers[-1].replace(',', '')

    return None

# ==========================================
# Examples of usage
# ==========================================
if __name__ == "__main__":
    test_cases = [
        # Case 1: Standard GSM8K format
        ("Janet has 5 apples. #### 5", "5"),
        
        # Case 2: Standard with currency and text
        ("Calculation: 50 + 50 = 100. #### $100 dollars", "100"),
        
        # Case 3: Standard with commas
        ("The total is huge. #### 1,234,567", "1234567"),
        
        # Case 4: No delimiter, but uses LaTeX boxed (common fallback)
        ("The answer is \\boxed{42}", "42"),
        
        # Case 5: Messy output (fallback to last number)
        ("I think the answer is 10. No wait, 5 + 5 is 10.  10.", "10"),
        
        # Case 6: Negative and decimals
        ("The temperature dropped. #### -5.5", "-5.5"),
    ]

    print(f"{'Input Snippet':<40} | {'Extracted':<10} | {'Status'}")
    print("-" * 65)
    
    for inputs, expected in test_cases:
        result = extract_gsm8k_answer(inputs)
        status = "✅ PASS" if result == expected else f"❌ FAIL (Got {result})"
        # Truncate input for display
        display_input = (inputs[:37] + '...') if len(inputs) > 37 else inputs
        print(f"{display_input:<40} | {str(result):<10} | {status}")