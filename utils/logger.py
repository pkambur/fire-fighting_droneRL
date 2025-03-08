import logging

logger_file = "./logs/program.log"

logging.basicConfig(filename=logger_file, level=logging.INFO,
                    format='%(asctime)s, %(message)s', filemode='w')
