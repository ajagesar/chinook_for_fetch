from utils.helpers import config, ROOT
from src.extraction import main as extraction
from src.preprocessing_presplit import main as preprocess_presplit
from src.splitting import main as splitting
from src.preprocessing_postsplit import main as preprocess_postsplit
from src.hyperparameter_tuning import main as hyperparameter_tuning


def main():


    if config['pipeline']['extraction'] == True:
        extraction()

    if config['pipeline']['preprocessing_presplit'] == True:
        preprocess_presplit()

    if config['pipeline']['splitting'] == True:
        splitting()

    if config['pipeline']['preprocessing_postsplit'] == True:
        preprocess_postsplit()

    if config['pipeline']['hyperparameter_tuning'] == True:
        hyperparameter_tuning()


    return


if __name__=="__main__":
    main()