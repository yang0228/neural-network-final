#Get timestamp for output
from datetime import datetime
def get_timestamp():
    raw = datetime.now()
    return str(raw.year).zfill(4) + str(raw.month).zfill(2) + str(raw.day).zfill(2) + "-" + str(raw.hour).zfill(2) + str(raw.minute).zfill(2)