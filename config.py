# configuration class
import torch


class CFG:
    loss_module = 'ArcFace'
    DATA_DIR = 'data/folds.csv'
    TEST_DIR = 'TESTDIR'
    TRAIN_DIR = 'TRAIN_DIR/train_images'
    seed = 123
    img_size = 512
    classes = 11014
    fc_dim = 512
    epochs = 1
    batch_size = 12
    num_workers = 3
    model_name = 'tf_efficientnet_b3'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path_arcface = 'model/Train_F1_score_0.9.pt'
    embedding_path = 'TrainedEmbeddings/training_image_embeddings_f1_0.9_arcface.npy'
    isTraining = True
    MODEL_PATH = 'model/'
    scheduler_params = {
        "lr_start": 3e-4,  # 1e-5
        "lr_max": 1e-5 * batch_size,  # 1e-5 * 32 (if batch_size(=32) is different then)
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }
    OUT_DIR = 'out'
    STREAM_LIT_DATA = 'streamlit_out'
