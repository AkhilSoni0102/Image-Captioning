# Image-Captioning
## Introduction:
  This was the project I did in my 6th semester as a final project for one course. We were a team of 3 people and had to do this in 1 month. The task is to generate a caption for the given image. I have used an encoder-decoder model to tackle this problem. The encoder model will combine the encoded form of image and text we got from CNN and RNN respectively and then the vectors resulting from both the encodings are then merged and processed by a Dense layer to make a final prediction. To encode our image feature we used the transfer learning method and to encode the text sequence we used the pre-trained Glove model.

## Understanding the dataset:
We have used flickr 8k dataset, each image is associated with five different captions that describe the entities and events depicted in the image. We chose flickr 8k because flickr 8k is a good starting dataset as it is small in size and can be trained easily on low-end laptops/desktops using a CPU.

## Data loading and preprocessing:
  After loading the data set, we’ll create a dictionary named “description” which contains the name of the image as key and a list of 5 captions for that image. Now traverse in this dictionary and perform some basic text cleaning like removing punctuations and converting everything to lowercase.

  Creating vocabulary of all unique words present in 8000*5 image captions in the data set and storing them in a set named “vocabulary”. We got 8828 unique words across 40000 image captions.

  Now save training and testing images in train_img and test_img lists. We load the description of the training images into a dictionary and will add two token ‘start_seq’ and ‘end_seq’ in every training captions

  We saw we have 8828 unique words i.e our vocabulary will be of size 8828, to make our model more robust we will reduce our vocabulary to only those words which occur at least 10 times in the entire corpus. This will reduce the vocabulary size to 1659. We also need to find out what the max length of a caption can be since we cannot have captions of arbitrary length. Max length of description is 38. We will create 2 dictionaries to map word to index and index to word. This indexing will be helpful when we will convert a word to its numerical representation using word embeddings.

## Word Embedding:
  Count vectorizer: each word in sentence is mapped to frequency of occurrence of it in the data on which the countvectorizer id trained Tfidf vectorizer: There are some words which are presents many times in the data that they start acting as stop words, giving them high value(freq) is of no sense as they are not giving any uniqueness to the representation as compared to other sentence, hence inverse document will decrease these of frequencies.

## Word2Vec:
  Words are in one hot encoding representation, an embedding matrix E is defined with its values set randomly, E*OC1 = ec1, in this way we will get the rest of the word's representations too. Now all these eci will be flattened and then using neural networks will train it and update the embedding matrix. And to get a word representation just multiply E to OHEc1. c1 c2 c3 c4 c5 CBOW: c1, c2, c4, c5 will predict c3 while training Skip Gram: c2 will be predicting c1, c3

## Glove embedding:
  e1= EOHEc1, e2= EOHEc2
  Now the embedding e1 and e2 we got, so while training, this will focus more on similarity(e1, e2) Cij= e1. e2= no of time e1 and e2 occurs together(co-occurrence). Glove focuses on co-occurrences of the word I with every other word in the corpus.

  The advantage of using Glove over Word2Vec is that GloVe does not just rely on the local context of words but it incorporates global word co-occurrence to obtain word vectors.

  We have used pre-trained glove embedding from glove.6B.200d which will give a 200 dimension vector. For each word and these words to vector, the representation will get stored in the embedding_index dictionary.

  Next, we make the matrix of shape(1665, 200) consisting of our vocabulary and the 200D vector.

## Model Building and Training:
  As you have seen from our approach we have opted for transfer learning using the InceptionV3 network which is pre-trained on the ImageNet dataset. We must remember that we do not need to classify the images here, we only need to extract an image vector for our images. Hence we remove the softmax layer from the inceptionV3 model. Since we are using InceptionV3 we need to pre-process our input before feeding it into the model. Hence we define a preprocess function to reshape the images to (299 x 299) and feed them to the preprocess_input() function of Keras.

preprocess_input() in Keras will transform the input to the required input in which the algorithm needs it. “Process” function is used to process the image → convert the image to an array → apply preproces_input() of Keras “Encode” function will give the feature vector of image from the model.predict(img)
