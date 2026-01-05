from utils.helpers import config, ROOT
from src.extraction import main as extraction
from src.preprocessing_presplit import main as preprocess_presplit
from src.splitting import main as splitting
from src.preprocessing_postsplit import main as preprocess_postsplit


def main():


    if config['extraction'] == True:
        extraction()


    if config['preprocessing_presplit'] == True:
        preprocess_presplit()

    if config['splitting'] == True:
        splitting()

    if config['preprocessing_postsplit'] == True:
        preprocess_postsplit()


    return


if __name__=="__main__":
    main()