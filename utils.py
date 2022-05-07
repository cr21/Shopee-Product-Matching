# Load trained model
from config import CFG
from arcface import ShopeeEncoderBackBone
import torch
import numpy as np
from sklearn.metrics import f1_score
import cv2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from dataset import ShopeeQueryDataset
import gc

def getPretrainedModel(loss_module='ArcFace', model_path=CFG.model_path_arcface, device=CFG.device):
    """
    Method to get pretrained model from file path


    Args:
        loss_module (string):  loss function Arcface, softmax
        model_path (nn.Modiule):  Model path
        device (torch.device):  cpu or gpu based on cuda enabled or not

    Returns:

        loaded model

    """
    if loss_module == 'ArcFace':
        # load arcface loss classfier
        model = ShopeeEncoderBackBone(isTraining=False)
        model.load_state_dict(torch.load(model_path, map_location=CFG.device))
        model = model.to(device)
        return model


def getTrainingEmbeddingsFromFile(filepath=CFG.embedding_path, max_count=None):
    """


    Args:
        filepath (): Embedding path
        max_count (): Maximum number of training images embedding we want

    Returns:

        return Matrix shape (N, 1536) N = number of training images embedding we want

    """
    embeddings = np.load(filepath)

    if max_count is None:
        return embeddings
    else:
        return embeddings[:max_count]


def get_images_path(root_dir):
    """

    Args:
        root_dir (): Dir containing all the images [ in our case only testing images]

    Returns:
        return list of all the images full absolute path
    """
    test_images_list = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f))]
    return test_images_list


# get nearest neighbors distances and index information

def get_knn_model(embeddings, KNN=50, metric='cosine'):
    knnModel = NearestNeighbors(n_neighbors=KNN, metric=metric)
    knnModel.fit(embeddings)

    return knnModel


def finding_thresholds_for_nearest_neighbors(df, embeddings, KNN=50, isImage=False, metric_param='cosine'):
    print(embeddings.shape)
    model = NearestNeighbors(n_neighbors=KNN, metric=metric_param)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    print(distances.shape)
    if not CFG.isTesting:

        # we will use different threshold for neighbor retrieval, basically we also want to make threashold smaller
        # to retrieval small number of records because we have testing image size around 70000 huge memory constraint
        if isImage:
            thresholds = list(np.arange(0.01, 4, 0.5))
        else:
            thresholds = list(np.arange(0.1, 1, 0.1))
        scores = []
        # for each threshold get top k neighbors with in threshold distance
        # then get f1 score
        for threshold in thresholds:
            predictions = []
            for k in range(embeddings.shape[0]):
                idx = np.where(distances[k,] < threshold)[0]
                ids = indices[k, idx]
                # get posting ids based on retrival set
                posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
                predictions.append(posting_ids)
            df['pred_matches'] = predictions
            df['f1'] = f1_score(df['matches'], df['pred_matches'])
            score = df['f1'].mean()
            print(f'Our f1 score for threshold {threshold} is {score}')
            scores.append(score)
        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'Our best score is {best_score} and has a threshold {best_threshold}')

        # Use threshold
        predictions = []
        print("for training time")
        for k in range(embeddings.shape[0]):
            # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
            if isImage:
                #                 print("choosing 0.2")
                idx = np.where(distances[k,] < best_threshold)[0]
            else:
                idx = np.where(distances[k,] < best_threshold)[0]
            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)

    # Because we are predicting the test set that have 70K images and different label groups, confidence should be smaller
    else:
        predictions = []
        for k in tqdm(range(embeddings.shape[0])):
            if isImage:
                # testing for different threshold after submission
                #                 idx = np.where(distances[k,] < 0.21 )[0]
                idx = np.where(distances[k,] < 0.3)[0]
            else:
                #                 idx = np.where(distances[k,] < 0.30)[0]
                idx = np.where(distances[k,] < 0.17)[0]
            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)

    del model, distances, indices
    gc.collect()
    return df, predictions


def get_neighbors(train_embeddings, query_embeddings, KNN=50, metric_param='cosine'):
    # we can get top neighbors based on different distance metric,in our case we are using
    # cosine and euclidean metric
    if metric_param == 'cosine':
        # fit cosine distance medal on train image embddings
        cosine_knnModel = get_knn_model(train_embeddings, KNN=KNN, metric='cosine')
        # get top k neighbors distances and indices given metric for query embeddings
        distances, indices = cosine_knnModel.kneighbors(query_embeddings)

    else:
        # fit euclidean distance modal on image embeddings
        eucl_knnModel = get_knn_model(train_embeddings, KNN=KNN, metric='minkowski')
        # get top k neighbors distances and indices given metric for query embeddings
        distances, indices = eucl_knnModel.kneighbors(query_embeddings)

    return distances, indices


def getEmbeddingsForSingleQuery(shopee_model, singleImagePath, transform):
    """
    Args:
        model (): already trained model
        singleImagePath (): Image Path String (absolute path)
        transform (): Transformation for model

    Returns:
        features from model (Embedding ) shape : (1,1536)

    """
    # read image from path
    image = cv2.imread(singleImagePath)
    # opencv read images in form of BGR so convert back to RGB form
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply transformation ( resizing and convert into pytorch tensor

    aug = transform(image=image)
    image = aug['image']
    label = torch.Tensor(1)
    # squeeze to add batch dimension before passing to pytorch model
    image = image.unsqueeze(0)

    shopee_model.eval()
    features = None
    with torch.no_grad():
        image = image.to(CFG.device)
        label = label.to(CFG.device)
        # forward pass to get features
        features = shopee_model(image, label)
    print(features.shape)
    return features


def getEmbeddingsforImagePathList(queryImagesPath, model, transform):
    """
    Args:
        queryImagesPath ():  List of Path of testing images
        model ():  trained model to get imbeddings
        transform (): test transformation before forwarding to model

    Returns:

        Embeddings of shape (len(queryImagesPath), 1536)

    """
    # create dataset from image paths
    query_dataset = ShopeeQueryDataset(queryImagesPath, transform=transform)

    # create dataloader
    query_dataloader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=16
    )

    # put model in evaluation mode
    model.eval()
    embeddings = []
    with torch.no_grad():
        for idx, datax in tqdm(enumerate(query_dataloader)):
            image, label = datax
            image = image.to(CFG.device)
            label = label.to(CFG.device)
            # forward pass to get features
            features = model(image, label)
            image_embeddings = features.detach().cpu().numpy()
            embeddings.append(image_embeddings)

    image_embeddings = np.concatenate(embeddings)

    return image_embeddings


def plot_canvas(train, COLS=4, ROWS=2, path=CFG.TRAIN_DIR + "/", img_list=[], k=0):
    for m in range(ROWS):
        _fig = plt.figure(figsize=(20, 5))
        for j in range(COLS):
            if j == 0 and m == 0:
                title = "Query Image \n"
                # title += "Downloaded from internet : \n"
                img = cv2.imread(img_list[k])
            else:
                row = COLS * m + j
                name = train.iloc[row - 1, 1]
                img = cv2.imread(path + name)

                title = "Recommended Image {} \n".format(row - 1)
                # orig_title = train.iloc[row-1,3]
                # punctuation= '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
                # for x in punctuation:
                #     orig_title=orig_title.replace(x,"")
                # title  += "title :" + orig_title[:min(15, len(orig_title))] + " \n"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, COLS, j + 1)
            plt.title(title)
            plt.axis('off')
            # plt.savefig("out/plot21_{}.png".format(j))
            plt.imshow(img)
        plt.show()
