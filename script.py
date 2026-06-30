# In your Streamlit app or script
ocr_json, qa_pairs = process_pdf("your_document.pdf")

# Save outputs
save_outputs(ocr_json, qa_pairs, base_name="result")

# Or use directly
for pair in qa_pairs:
    print(f"Q: {pair['question']}")
    print(f"A: {pair['answer'][:200]}...")
    print("-" * 50)
