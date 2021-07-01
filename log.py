import logging
import inspect
from datetime import datetime

def get_lgger(level):
    # create logger object and set level with input level mentioned
    logger=logging.getLogger('SystemLogging')
    logger.setLevel(level)
    #create handler object
    #Filename=inspect.stack()[1][3]                        # it will give you the function name calling this code
    #print(Filename)
    Filename='logs/SystemLog'    #+str(datetime.now().strftime('%d%m%Y%H%M%S'))
    handler=logging.FileHandler(filename='{}.log'.format(Filename),mode='w') # creating log file with caler nameu
    handler.setLevel(level)
    #create formatter object
    formatter=logging.Formatter('%(asctime)s-%(message)s')
    #set formatter to handler object
    handler.setFormatter(formatter)
    #add handler to logger object
    logger.addHandler(handler)
    return logger