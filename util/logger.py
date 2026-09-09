import os
import logging
from datetime import datetime


def get_logger(mode, workdir):
    # get logger
    logger = logging.getLogger("Logger_{}".format(mode))

    # log filename
    log = workdir + "/log.log"

    # log formation
    formatter = logging.Formatter("[%(asctime)s;%(levelname)s]%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)

    filehandler = logging.FileHandler(log)
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    logger.setLevel(logging.INFO)
    return logger
