import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(logging.DEBUG)
# stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler("logs/logs.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
# Xóa tất cả handlers đã có (nếu có)
if logger.hasHandlers():
    logger.handlers.clear()


logger.addHandler(file_handler)
# logger.addHandler(stdout_handler)
