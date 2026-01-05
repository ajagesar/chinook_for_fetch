from utils.helpers import config, ROOT
from src.extraction import main as extraction
from src.extraction import main as preprocess_presplit

def main():

    if config['extraction'] == True:
        extraction()

    if config['preprocessing_presplit'] == True:
        preprocess_presplit()

    return


if __name__=="__main__":
    main()