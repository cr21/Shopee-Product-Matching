
# Shopee-Product-Matching

![Shopee](/Shopee%20Images/Shopee_image1.jpeg)


## Business Problem

Shopee is the leading e-commerce platform in Southeast Asia and Taiwan. Customers appreciate its easy, secure, and fast online shopping experience tailored to their region. The company also provides strong payment and logistical support along with a 'Lowest Price Guaranteed' feature on thousands of Shopee's listed products.

Finding near-duplicates in large datasets is an important problem for many online businesses. In Shopee's case, everyday users can upload their own images and write their own product descriptions, adding an extra layer of challenge. Your task is to identify which products have been posted repeatedly. The differences between related products may be subtle while photos of identical products may be wildly different!

Two different images of similar wares may represent the same product or two completely different items. Retailers want to avoid misrepresentations and other issues that could come from conflating two dissimilar products. Currently, a combination of deep learning and traditional machine learning analyzes image and text information to compare similarity. But major differences in images, titles, and product descriptions prevent these methods from being entirely effective.

In this competition, we’ll apply our machine learning skills to build a model that predicts which items are the same products.

## Understanding the data
[train/test].csv - the training set metadata. Each row contains the data for a single posting. Multiple postings might have the exact same image ID, but with different titles or vice versa.
    
- posting_id - the ID code for the posting.
- image - the image id/md5sum.
- image_phash - a [perceptual hash](https://en.wikipedia.org/wiki/Perceptual_hashing) of the image.
- title - the product description for the posting.
- label_group - ID code for all postings that map to the same product. Not provided for the test set

[train/test]images - the images associated with the postings.
sample_submission.csv - a sample submission file in the correct format.

- posting_id - the ID code for the posting.
- matches - Space delimited list of all posting IDs that match this posting. Posts always self-match. Group sizes were capped at 50, so there's no need to predict more than 50 matches.

## Workflow
<!-- ![workflow](/Shopee%20Images/Model_Shopee.jpeg) -->
![workflow2](/Shopee%20Images/workflow_model.jpeg)


## Overall Basic pipeline

![CBIR](/Shopee%20Images/CBIR.jpeg) 

## High Level Solution Design 

We divide the overall solution in the following 3 parts.

1. **Learn to represent objects ( Product titles, product images) as a continuous dense vector** ( i.e. generate product title embeddings and product image embeddings)
2. **Learn to place similar objects together.** Similar product image should be in neighborhood, same way similar product title should be in same neighborhood. More difficult task would be to place a similar product title and similar product image in the same neighborhood. (Our next step)
3. Learned to **retrieve neighboring** objects and product embeddings **really fast.**

Let's simplify above steps further :

### Classification CNN
We were given information that **all the similar product have same label group**, We can leverage this information to build classification model to classify images into label group. 

we train a classification CNN by inputting product images and getting a one hot vector output that represents the label group of the image. For example, pretend we train a CNN to classify ten types of product items and input an image. Then the one-hot-vector output pictured below predicts product label group 4.

![cnn_embeddings](/Shopee%20Images/cnn_embeddings.jpeg)

### Generating Embeddings 

Now imagine that we want to compare two product images (of product that are not one of the  label groups, or with in the label groups) and decide whether they are similar. Images are hard to compare, but numbers are easy to compare. So we input an image into a CNN and take the activations of the last layer before output layer, which we called dense representation of image aka image embedding. In the picture above that is a vector of dimension 64. So we can input two images, get two embeddings, and then compare the embeddings. The CNN embeddings are meaningful because they represent patterns that are detected in the images.

### Cosine Distance
We compare vectors (numbers) by computing the distance between them. What is the distance between the 3-dimensional vector [0.2, 0.9, 0.7] and [0.5, 0.4, 0.1]?

There is no right answer because there are many ways to calculate distance. This problem further can be solved using metric learning approach, I will try to explore this in my future work. In high school we learn Euclidean distance, then the answer would be sqrt( (0.5-0.2)**2 + (0.4-0.9)**2 + (0.1-0.7)**2 ). If you imagine the vectors as points in 3-space, then Euclidean distance is literally the distance between them.

And **cosine distance** would be **one minus the cosine of the angle from point one to the origin to point two.** 
**This equals 0 when the points are the same, and 1 when the points are far away.**




## List Of Approaches I tried


1. [Convolution AutoEncoder - BaseLine](/AutoEncoder%5BBaseline%5D.ipynb)
    ### Apparoach
    AutoEncoder model consist of two parts, Encoder  and Decoder. Encoder downsamples the image to lower dimension dimension features, and decoder is used to reconstruct the same image using latent dimension.
    
    My Hypothesis is that If I am able to regenerate the same image with with little error, then we can say that latent dimension is compressed and dense feature that captures the image information in lower dimension.
    
    After training model, we will pass all images to encoder to generate the latent features, we will store latent features to database. At test time, we will pass image to encoder to get query features. we will then compute the euclidean distance to all the features in database to get top predction. 
    
    ### Result:  
    AutoEncoder produce the decent result, but it is still not good approach to generate the semantically similar image.
    
    it is also error prone and give some useless result, in AutoEncoder we rely on MSE loss which will focus on reducing each pixel error distance, which is misleading in semantic similarity.
    
2. [Label Group MultiClass classification using Weighted Random Sampler using multiclass Cross Entropy loss ](/PriceLabelClassification%5BTraining%5D.ipynb)

    
    ### Solution Approach
    
    * We were given information that all the **similar product have same label group.** 
    * We can leverage this information to build **classification model to classify images into label group.**
    * From Image EDA, I found out that we have **11014** different classes, and dataset is **not balanced dataset**, If you see below plot, we can clearly see that there are **hardly 1000 data points having more than 10 products per label.**
    * In this notebook I used **Weighted Sampler technique used in pytorch for handling imbalanced classification problem**

    ![Label_freq](/Shopee_Repo_Images/Label_frequency_plot.png)
    
    ### Results
    
    * Using **Weighted Sampler technique** really helped me to **improve classification accuracy** for **under represented label groups ( label groups for which only 2 product images** were available.
    * I achivied **0.62 F1 Score** which is significant improvement from earlier baseline model.
    

