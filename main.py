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
        print("\n" + "="*50)
        print("STARTING: Data Extraction")
        print("="*50 + "\n")
        extraction()
        print("\n✓ COMPLETED: Data Extraction\n")
    
    if config['pipeline']['preprocessing_presplit'] == True:
        print("\n" + "="*50)
        print("STARTING: Preprocessing (Pre-Split)")
        print("="*50 + "\n")
        preprocess_presplit()
        print("\n✓ COMPLETED: Preprocessing (Pre-Split)\n")
    
    if config['pipeline']['splitting'] == True:
        print("\n" + "="*50)
        print("STARTING: Data Splitting")
        print("="*50 + "\n")
        splitting()
        print("\n✓ COMPLETED: Data Splitting\n")
    
    if config['pipeline']['preprocessing_postsplit'] == True:
        print("\n" + "="*50)
        print("STARTING: Preprocessing (Post-Split)")
        print("="*50 + "\n")
        preprocess_postsplit()
        print("\n✓ COMPLETED: Preprocessing (Post-Split)\n")
    
    if config['pipeline']['hyperparameter_tuning'] == True:
        print("\n" + "="*50)
        print("STARTING: Hyperparameter Tuning")
        print("="*50 + "\n")
        hyperparameter_tuning()
        print("\n✓ COMPLETED: Hyperparameter Tuning\n")
    
    if config['pipeline']['modeling'] == True: # PM if parallel processing error when modeling, first run pipeline until preprocessing_postsplit, then second time from modeling only (adjust settings in config file)
        print("\n" + "="*50)
        print("STARTING: Model Training")
        print("="*50 + "\n")
        modeling()
        print("\n✓ COMPLETED: Model Training\n")
    
    if config['pipeline']['evaluation'] == True:
        print("\n" + "="*50)
        print("STARTING: Model Evaluation")
        print("="*50 + "\n")
        evaluation()
        print("\n✓ COMPLETED: Model Evaluation\n")
    
    if config['pipeline']['plotting'] == True:
        print("\n" + "="*50)
        print("STARTING: Plotting Results")
        print("="*50 + "\n")
        plotting()
        print("\n✓ COMPLETED: Plotting Results\n")
    
    print("\n" + "="*50)
    print("Pipeline has finished.")
    print("="*50 + "\n")
    return

if __name__=="__main__":
    main()