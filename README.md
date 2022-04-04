
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
![workflow](/Shopee%20Images/Model_Shopee.jpeg)
![workflow2](/Shopee%20Images/workflow_model.jpeg)


## Overall Basic pipeline

![CBIR](/Shopee%20Images/CBIR.jpeg) 

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

    * In this competition it is given that,if two or more images have **same label group** then they are **similar products.** 
    * Basically we can use this information to transfer the business problem into **multi class classification** problem.
    * From Image EDA, I found out that we have **11014** different classes, and dataset is **not balanced dataset**
    * If you see below plot, we can clearly see that there are **hardly 1000 data points having more than 10 products per label.*
    * In this notebook I used **Weighted Sampler technique used in pytorch for handling imbalanced classification problem**

    ![Label_freq](/Shopee_Repo_Images/Label_frequency_plot.png)
    
    ### Results
    
    * Using **Weighted Sampler technique** really helped me to **improve classification accuracy** for **under represented label groups ( label groups for which only 2 product images** were available.
    * I achivied **0.62 F1 Score** which is significant improvement from earlier baseline model.
    

