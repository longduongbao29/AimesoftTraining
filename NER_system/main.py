# main.py
from nlp.text_processor import TextProcessor
from utils.file_handler import FileHandler


def main():
    # Đọc nội dung từ file
    input_path = "files/Ner.txt"
    output_path = "files/output.txt"

    content = FileHandler.read_file(input_path)

    # Xử lý nội dung
    processor = TextProcessor()
    output_content = processor.process_text(content)

    # Ghi nội dung đã xử lý vào file
    FileHandler.write_file(output_path, output_content)

    print(f"Processed text has been written to {output_path}")


if __name__ == "__main__":
    main()
