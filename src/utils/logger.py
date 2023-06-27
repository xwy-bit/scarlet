import logging
import os

# log config
logger = logging.getLogger('CHAT-LOG')
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)-5s: %(message)s")

# config logger output file
log_path = 'logs/test_pdf.log'
log_file_is_existed = os.path.exists('/'.join(log_path.split('/')[:-1])+'/')
if not log_file_is_existed:
    os.makedirs('/'.join(log_path.split('/')[:-1])+'/')
fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

# config logger to screen
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.info('='*100)
logger.addHandler(ch)