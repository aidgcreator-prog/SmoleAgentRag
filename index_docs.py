#!/usr/bin/env python3
"""
index_docs.py — CLI indexing tool for the Multipurpose AI Assistant (by LocalAiLab).
Indexes documents into the app's ChromaDB knowledge base without launching the UI.

Usage examples:
  # Index a PDF
  python index_docs.py --pdf path/to/doc.pdf

  # Index a text file
  python index_docs.py --txt path/to/notes.txt

  # Index a whole folder of PDFs + txts
  python index_docs.py --dir ./my_docs

  # Load a HuggingFace dataset
  python index_docs.py --hf-dataset m-ric/huggingface_doc --text-col text --source-col source

  # Show stats
  python index_docs.py --stats

  # Clear the index
  python index_docs.py --clear
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Index documents into the RAG knowledge base.")
    parser.add_argument("--pdf",         help="Path to a PDF file to index")
    parser.add_argument("--txt",         help="Path to a TXT/MD file to index")
    parser.add_argument("--docx",        help="Path to a DOCX file to index")
    parser.add_argument("--dir",         help="Directory; indexes all .pdf / .txt / .md / .docx files inside")
    parser.add_argument("--hf-dataset",  help="HuggingFace dataset name")
    parser.add_argument("--text-col",    default="text",   help="Column containing text (for HF datasets)")
    parser.add_argument("--source-col",  default="source", help="Column containing source name (optional)")
    parser.add_argument("--stats",       action="store_true", help="Print index stats and exit")
    parser.add_argument("--clear",       action="store_true", help="Clear the entire index and exit")
    args = parser.parse_args()

    # Import after argparse so --help is instant
    from app import (
        get_embed_model, get_chroma_collection, get_index_stats,
        index_pdf_file, index_txt_file, index_docx_file, index_hf_dataset, clear_index,
    )

    # Pre-warm
    get_embed_model()
    get_chroma_collection()

    if args.clear:
        # clear_index returns ([], message_string)
        _, msg = clear_index()
        print(msg)
        return

    if args.stats:
        print(get_index_stats())
        return

    if args.pdf:
        # index_pdf_file returns a plain string
        print(index_pdf_file(args.pdf))

    if args.txt:
        # index_txt_file returns a plain string
        print(index_txt_file(args.txt))

    if args.docx:
        # index_docx_file returns a plain string
        print(index_docx_file(args.docx))

    if args.dir:
        folder = Path(args.dir)
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() == ".pdf":
                print(index_pdf_file(str(f)))
            elif f.suffix.lower() in (".txt", ".md"):
                print(index_txt_file(str(f)))
            elif f.suffix.lower() == ".docx":
                print(index_docx_file(str(f)))

    if args.hf_dataset:
        # index_hf_dataset returns (message_string, table_list)
        msg, _ = index_hf_dataset(args.hf_dataset, args.text_col, args.source_col)
        print(msg)

    print(get_index_stats())


if __name__ == "__main__":
    main()
