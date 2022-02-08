
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

