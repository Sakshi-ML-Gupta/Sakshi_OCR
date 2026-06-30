# Direct use
ocr_json, qa_pairs = process_pdf("your_pdf.pdf")

# Check completeness
for pair in qa_pairs:
    if pair['matched']:
        print(f"Question: {pair['question'][:50]}...")
        print(f"Answer length: {len(pair['answer'])} characters")
        print(f"Answer preview: {pair['answer'][:200]}...")
        print("-" * 50)
