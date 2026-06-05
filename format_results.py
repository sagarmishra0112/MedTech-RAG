import csv
import ast
import os

input_file = r"d:\FinanceRAG\data\eval\ragas_results.csv"
output_file = r"d:\FinanceRAG\data\eval\ragas_results_formatted.md"

def format_csv_to_md():
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        
        f_out.write("# 📊 RAGAS Evaluation Results\n\n")
        
        for i, row in enumerate(reader, 1):
            f_out.write(f"## 📝 Test Case {i}\n\n")
            f_out.write(f"**🗣️ User Input (Question):**\n> {row.get('user_input', '').replace(chr(10), chr(10)+'> ')}\n\n")
            
            # Format contexts
            contexts = row.get('retrieved_contexts', '[]')
            f_out.write(f"**📚 Retrieved Contexts:**\n")
            try:
                if contexts.startswith('['):
                    ctx_list = ast.literal_eval(contexts)
                    for j, ctx in enumerate(ctx_list, 1):
                        f_out.write(f"*Chunk {j}:*\n```text\n{ctx.strip()}\n```\n\n")
                else:
                    f_out.write(f"```text\n{contexts}\n```\n\n")
            except Exception as e:
                f_out.write(f"```text\n{contexts}\n```\n\n")

            f_out.write(f"**🎯 Reference (Golden) Answer:**\n{row.get('reference', '')}\n\n")
            f_out.write(f"**🤖 RAG AI Response:**\n{row.get('response', '')}\n\n")
            
            
            f_out.write(f"### 📈 Metrics Scorecard\n")
            f_out.write(f"- **Context Precision:** `{row.get('context_precision', 'N/A')}`\n")
            f_out.write(f"- **Context Recall:** `{row.get('context_recall', 'N/A')}`\n")
            f_out.write(f"- **Faithfulness:** `{row.get('faithfulness', 'N/A')}`\n")
            f_out.write(f"- **Answer Relevancy:** `{row.get('answer_relevancy', 'N/A')}`\n\n")
            
            f_out.write("---\n\n")

    print(f"Results successfully formatted and saved to: {output_file}")

if __name__ == "__main__":
    format_csv_to_md()
