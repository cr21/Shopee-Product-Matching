from utils import getPretrainedModel, get_knn_model, get_neighbors, get_images_path, \
    getTrainingEmbeddingsFromFile, getEmbeddingsForSingleQuery, getEmbeddingsforImagePathList, plot_canvas
from config import CFG
from arcface import get_test_transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# load dataset
import os
import streamlit as st


def main():
    """Product Semantic Similarity Using Representation Learning"""

    st.title("Product Semantic Similarity Using Representation Learning")
    st.image(os.path.join(CFG.STREAM_LIT_DATA,'Shopee_image1.jpeg'))
    st.write("We divide the overall solution in the following 3 parts.\n"
                "1. Learn to represent objects ( Product titles, product images) as a continuous dense vector ( i.e. generate product title embeddings and product image embeddings).\n"
                "2. Learn to place similar objects together. Similar product image should be in neighborhood, same way similar product title should be in same neighborhood. More difficult task would be to place a similar product title and similar product image in the same neighborhood. (Our next step).\n"
                "3. Learned to retrieve neighboring objects and product embeddings really fast")

    st.subheader("Classification CNN")
    st.write("We were given information that all the similar product have same label group, We can leverage this information to build classification model to classify images into label group.")
    st.write("We train a classification CNN by inputting product images and getting a one hot vector output that represents the label group of the image. For example, pretend we train a CNN to classify ten types of product items and input an image. Then the one-hot-vector output pictured below predicts product label group 4.")
    st.image(os.path.join(CFG.STREAM_LIT_DATA, 'cnnEmbeddings.jpeg'))
    st.image(os.path.join(CFG.STREAM_LIT_DATA, 'vgg_embeddings.jpeg'))
    st.subheader("Generating Embeddings")
    st.write("Now imagine that we want to compare two product images (of product that are not one of the label groups, or with in the label groups) and decide whether they are similar. Images are hard to compare, but numbers are easy to compare. So we input an image into a CNN and take the activations of the last layer before output layer, which we called dense representation of image aka image embedding. After getting all the training product embeddings, we will get query embeddings and for every query embeddings we will get top k similar product  using KNN.")

    st.subheader("Arcface")
    st.write("We would like similar classes ( Product belongs to same label_group) to have  embeddings close to each other and dissimilar classes (Product belongs to different label_group) to be far from each other, But why would this happen? We didn't train our model to do this, we only trained our model to predict product accurately"
    "ArcFace adds more loss to the training procedure to encourage similar class embeddings to be close and dissimilar embeddings to be far from each other. This is adding a second task to the training of a CNN. The first task is predicting the image accurately.")
    st.image(os.path.join(CFG.STREAM_LIT_DATA, 'Arcafceloss.jpeg'))
    st.subheader("Loading data...")
    train_df = pd.read_csv('data/folds.csv')
    st.code("train_df = pd.read_csv('data/folds.csv')")
    if st.button("Show Data"):
        st.write(train_df.head())
        st.write("Shape of dataset: ",train_df.shape)


    st.subheader("We have trained our Model already and we will be using pre-trained model from file")
    # get pretrained Pytorch model from file
    st.code("shopee_model = getPretrainedModel(loss_module='ArcFace', model_path=CFG.model_path_arcface, device=CFG.device)")
    shopee_model = getPretrainedModel(loss_module='ArcFace',model_path=CFG.model_path_arcface, device=CFG.device)
    st.write("Shopee Model loaded successfully")

    #st.warning("Since the original data contains 35000 images, we will be subsetting 1000 images for improved performance")
    st.subheader("Getting training embeddings for images from file")
    st.code("training_embeddings = getTrainingEmbeddingsFromFile(CFG.embedding_path)")
    training_embeddings = getTrainingEmbeddingsFromFile(CFG.embedding_path)
    st.write("TrainingEmbedding shape {}".format(training_embeddings.shape))

    st.subheader("Getting K-nearest model from file")
    st.code("knn_model_cosine = get_knn_model(training_embeddings, KNN=50, metric='cosine')")
    knn_model_cosine = get_knn_model(training_embeddings, KNN=50, metric='cosine')
    st.write("KNN MODEL for cosine metric loaded successfully")

    st.subheader("Getting Query Imagespath")
    st.code("query_images_path = get_images_path(CFG.TEST_DIR)")
    query_images_path = get_images_path(CFG.TEST_DIR)
    #st.write(query_images_path)
    st.subheader("Getting query embeddings from list of paths")
    st.code("query_emebddings = getEmbeddingsforImagePathList(queryImagesPath=query_images_path,model = shopee_model,transform=get_test_transforms())")
    query_emebddings = getEmbeddingsforImagePathList(queryImagesPath=query_images_path,model = shopee_model,
                                                 transform=get_test_transforms())
    st.write("Query Embeddings loaded successfully Query Embedding shape {}".format(query_emebddings.shape))

    st.subheader("Getting top 5 [KNN=5] neighbors for all the query image embeddings from training embeddings")

    st.image(os.path.join(CFG.STREAM_LIT_DATA, 'Embedding_pic.png'))
    st.code(
        "query_cosine_distances, query_cosine_indices = get_neighbors(train_embeddings=training_embeddings,query_embeddings=query_emebddings,KNN=50, metric_param='cosine')")
    query_cosine_distances, query_cosine_indices = get_neighbors(train_embeddings=training_embeddings,
                                                                 query_embeddings=query_emebddings,
                                                                 KNN=50, metric_param='cosine')

    st.subheader("Top 50 similar training product indices successfully final indices shape : {}".format(
        query_cosine_indices.shape))
    st.write(query_cosine_indices)

    st.subheader("Let's see Some recommendations made by the model")
    indices = [i for i in range(7)]
    # k = st.selectbox("Select Indice of the testing image you want to see recommendations for", indices)
    # st.write(k)
    # st.write(query_cosine_distances[k,])
    for k in indices:
    # if st.button("Recommend based on the indice of the test image chosen above"):
        # plt.figure(figsize=(20,3))
        # plt.plot(np.arange(8),query_cosine_distances[k,][:8],'o-')
        # plt.title('Image {} Distance From Train Row {} to Other Train Rows'.format("cosine",k),size=16)
        # plt.ylabel('{} Distance to Train Row {}'.format("cosine", k),size=14)
        # plt.xlabel('Index Sorted by {} Distance to Train Row {}'.format("cosine",k),size=14)
        # plt.show()

        cluster = train_df.loc[query_cosine_indices[k, :5]]
        fig = plot_canvas(cluster, COLS=5, ROWS=1, path=CFG.TRAIN_DIR + "/", img_list=query_images_path, k=k)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)

    st.success("This was a deployment of our Project on Streamlit app. Below you can find some links for reference.")
    st.write(
        "Check out this [Image reference](https://www.kdnuggets.com/2019/01/building-image-search-service-from-scratch.html)")
    st.write("Check out this [Arcface Research Paper](https://arxiv.org/pdf/1801.07698.pdf)")


if __name__ == '__main__':
    main()



