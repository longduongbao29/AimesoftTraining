# utils/file_handler.py
class FileHandler:
    @staticmethod
    def read_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def write_file(file_path, content):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
