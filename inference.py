import sys

from utils import getPretrainedModel, get_neighbors, get_images_path, \
    getTrainingEmbeddingsFromFile, getEmbeddingsforImagePathList, plot_canvas
from config import CFG
from arcface import get_test_transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Training Mode off
    CFG.isTraining = False
    train_df = pd.read_csv(CFG.DATA_DIR)

    # Load Trained Model
    shopee_model = getPretrainedModel(loss_module='ArcFace', model_path=CFG.model_path_arcface, device=CFG.device)

    # we already stored Training Embeddings in files, so Let's use it
    training_embeddings = getTrainingEmbeddingsFromFile(CFG.embedding_path)

    # If you don't have training embedding saved and want to generate it uncomment below line
    # train_image_paths = get_images_path(CFG.TRAIN_DIR)
    # training_embeddings = getEmbeddingsforImagePathList(queryImagesPath=train_image_paths, model=shopee_model,
    #                                                     transform=get_test_transforms())

    # get Query image path
    query_images_path = get_images_path(CFG.TEST_DIR)
    # get query embeddings
    query_emebddings = getEmbeddingsforImagePathList(queryImagesPath=query_images_path, model=shopee_model,
                                                     transform=get_test_transforms())
    # get cosine distance and cosine indices to top 50 neighbors
    query_cosine_distances, query_cosine_indices = get_neighbors(train_embeddings=training_embeddings,
                                                                 query_embeddings=query_emebddings,
                                                                 KNN=50, metric_param='cosine')

    indices = [i for i in range(6)]
    for k in indices:
        plt.figure(figsize=(20, 3))
        plt.plot(np.arange(8), query_cosine_distances[k,][:8], 'o-')
        plt.title('Image {} Distance From Train Row {} to Other Train Rows'.format("cosine", k), size=16)
        plt.ylabel('{} Distance to Train Row {}'.format("cosine", k), size=14)
        plt.xlabel('Index Sorted by {} Distance to Train Row {}'.format("cosine", k), size=14)

        cluster = train_df.loc[query_cosine_indices[k, :5]]
        fig = plot_canvas(cluster, COLS=5, ROWS=1, path=CFG.TRAIN_DIR + "/", img_list=query_images_path, k=k)


if __name__ == '__main__':
    main()
