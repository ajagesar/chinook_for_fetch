from utils.helpers import config, ROOT
from src.extraction import main as extraction

def main():

    if config['extraction'] == True:
        extraction()

    

    return


if __name__=="__main__":
    main()