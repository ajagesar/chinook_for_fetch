from utils.helpers import config, ROOT
from src.extraction import main as extraction
from src.preprocessing_presplit import main as preprocess_presplit
from src.splitting import main as splitting
from src.preprocessing_postsplit import main as preprocess_postsplit
from src.hyperparameter_tuning import main as hyperparameter_tuning
from src.modeling import main as modeling
from src.evaluation import main as evaluation
from src.plotting import main as plotting


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

    if config['pipeline']['modeling'] == True:
        modeling()

    if config['pipeline']['evaluation'] == True:
        evaluation()

    if config['pipeline']['plotting'] == True:
        plotting()
        

    return


if __name__=="__main__":
    main()