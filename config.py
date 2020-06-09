import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--infile', type=str, default=u"", help='Data path.')
    parse.add_argument('--Single', type=bool, default=True, help='Single/Multi load.')
