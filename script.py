ocr_json, qa_pairs = process_pdf("your_file.pdf")

for pair in qa_pairs:
    if pair['matched']:
        print(f"Question: {pair['question'][:50]}...")
        print(f"Answer length: {len(pair['answer'])} characters")
        print(f"First 200 chars: {pair['answer'][:200]}")
        print(f"Last 200 chars: {pair['answer'][-200:]}")
        print("=" * 50)
