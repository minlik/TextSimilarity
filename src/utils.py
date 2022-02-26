from datetime import datetime


def print_bar():
    now_time = datetime.now()
    print('========' * 8 + now_time.strftime('%H:%M:%S'))
