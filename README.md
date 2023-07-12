<a name="br1"></a> 

A project report on

**IMAGE RESTORATION**

Submitted in partial fulfillment of the requirements for the Degree of

B. Tech in Computer Science & Engineering

by

**Kush Jayank Pandya (1705701)**

**Biswajeet Sahoo (1705689)**

**Daibik DasGupta (1705692)**

under the guidance of

**Manjusha Pandey**

School of Computer Engineering

Kalinga Institute of Industrial Technology

Deemed to be University

Bhubaneswar

November 2020



<a name="br2"></a> 



<a name="br3"></a> 

**CERTIFICATE**

This is to certify that the project report entitled **“Image Restoration”** submitted by

**Kush Jayank Pandya**

**Biswajeet Sahoo**

**1705701**

**1705689**

**1705692**

**Daibik DasGupta**

in partial fulfillment of the requirements for the award of the **Degree of Bachelor of**

**Technology** in **Discipline of Engineering** is a bonafide record of the work carried out

under my(our) guidance and supervision at School of Computer Science, Kalinga

Institute of Industrial Technology, Deemed to be University.

(Signature)

Manjusha Pandey

KIIT School of Computer Science

Kalinga Institute of Industrial Technology

................................................................................................................................................

**The Project was evaluated by us on 25/11/2020**

Manjusha Pandey



<a name="br4"></a> 

ACKNOWLEDGEMENTS

It is with a sense of satisfaction that our group is able to complete and compile this

project. However, it would not have been possible without the aid from our project guide.

We would like to express our gratitude to Prof. Manjusha Pandey, whose advice and

guidance has proven invaluable in bringing about the end product. I would also go on to

acknowledge the support and aid that was given to us by the Dean of our School of

Computer Engineering, Prof. Bhabani Shankar Prasad Mishra and our Director Prof.

Samaresh Mishra. It is with the help from all of these individuals that we have been able

to see our project through to completion.

**Kush Jayank Pandya**

**Biswajeet Sahoo**

**Daibik DasGupta**



<a name="br5"></a> 

ABSTRACT

It is not uncommon for image data to be corrupted due to unforeseen factors. There are a

number of reasons which render the photos corrupt such as accumulation of bad sectors

on the storage media, some bits missing, scratch on CDs/DVDs, and so on. Not to

mention the issues of the camera itself that may occur causing noise in the image due to

another variety of reasons. While there are preventative solutions to this, the restoration

of these images is not so often seen. Even if they are seen, usually it is a time consuming

and arduous process. This predicament of image restoration is what we are attempting to

solve through our project. We achieve this through deep learning models. But not only

are we attempting to restore the original image once it has been corrupted, we are also

training the model to be the most efficient model possible that can achieve this. Our final

models obtained are two multi-layer deep learning models, optimized further by the

appropriate loss functions to produce very accurate and clear images as output from

noised input images.

1



<a name="br6"></a> 

TABLE OF CONTENTS

**Abstract**

1

**Table of Contents**

2

**List of Figures**

4

**List of Tables**

5

1

**Introduction**

6

1\.1

1\.2

1\.3

2

Problem Statement

6

Project Description

Future Scope

6

6

**Background**

7

2\.1

2\.2

3

Reading Material

7

Related Theory/Terms

**Project Analysis/Implementation**

Technology Used and Basic Framework

Details of Deep Learning Model

7

9

3\.1

3\.2

9

9

3\.2.1 Workflow

9

3\.2.2 Noise

10

11

11

12

14

14

16

18

20

22

24

27

27

27

28

28

29

30

3\.2.3 Data Augmentation

3\.2.4 Self-Supervising Nature

3\.2.5 Loss Function

**Results & Discussions**

Small Model 1

4

4\.1

4\.2

4\.3

4\.4

4\.5

4\.6

5

Small Model 2

Normal Model 1

Normal Model 2

Enhanced Model 1

Enhanced Model 2

**Conclusion and Future Work**

Conclusion

5\.1

5\.2

5\.3

Future Work

Planning and Project Management

**References**

**Individual Contribution Report**

**Plagiarism Report**

2



<a name="br7"></a> 

LIST OF FIGURES

**Figure**

**ID**

**Figure Title**

**Page**

2\.1

2\.2

2\.3

2\.4

3\.1

3\.2

3\.3

4\.1

5\.1

Convolutional Neural Network

7

8

Image Noised and Denoised

Loss Function Variations

Data Augmentation Chart

Planning Model

8

8

10

11

14

25

27

Noise Chart

Model 1 and 2 Figures

Denoising Observation

Gantt Chart

3



<a name="br8"></a> 

LIST OF TABLES

**Table**

**ID**

**Table Title**

**Page**

4\.1

4\.2

Loss Function Table for Small Model 1

13

13

15

15

17

17

19

19

21

21

23

23

26

Small Model 1 Results

4\.3

Loss Function Table for Small Model 2

Small Model 2 Results

4\.4

4\.5

Loss Function Table for Model 1

Model 1 Results

4\.6

4\.7

Loss Function Table for Model 2

Model 2 Results

4\.8

4\.9

Loss Function Table for Enhanced Model 1

Enhanced Model 1 Results

Loss Function Table for Enhanced Model 2

Enhanced Model 2 Results

Schedule

4\.10

4\.11

4\.12

5\.1

4



<a name="br9"></a> 

**CHAPTER 1**

**INTRODUCTION**

Tells the reader what the report is about, aims and objectives of the project work

and how the report is organized into different chapters.

1\.1 Problem Statement

It is not uncommon for image data to be corrupted due to unforeseen factors. There

are a number of reasons which render the photos corrupt such as accumulation of

bad sectors on the storage media, some bits missing, scratch on CDs/DVDs, and so

on.[1] Not to mention the issues of the camera itself that may occur causing noise in

the image due to another variety of reasons. While there are preventative solutions

to this, the restoration of these images is not very commonly utilized or seen.

1\.2 Project Description

We are working with a small dataset of images where we apply a number of

different noises on the image, then we attempt to introduce these corrupted images

to the model and train it to adapt to the various different possible noises that it may

encounter in its life cycle. This is how we are producing a self supervising model

that can deal with any kind of image corruption and achieve versatility and

efficiency.

This project is being made as a major project by KIIT students. This is a Machine

Learning Project. We have collected the Dataset of images, apply a number of

different types of noise on the images and then use our model to restore the image.

By growing our model we have attempted to find the best possible model we could.

1\.3 Project Report

The project report will initially give an insight into the problem statement and why

it is important for the issue to be resolved and it will give clarity on our approach to

this issue and how we differentiate from other approaches. Finally, we will provide

the results obtained from our hard work.

5



<a name="br10"></a> 

**CHAPTER 2**

**BACKGROUND**

2\.1 Reading Material

The following medical research and machine learning repositories were studied to

form a basis of the solution for our problem statement.

1\.) “Introduction to image denoising” by Nabil Madali

2\.) Image Restoration, Computer Vision

3\.) Multi-Scale Structural Similarity Index for Image Quality

4\.) Stanford Engineering, CS, Image Processing and All You Need to Know, Prof.

Wang Lee, Prof. Elliot Junes

2\.2 Related Theory/Terms

**Deep learning** is an artificial intelligence (AI) function that imitates the workings

of the human brain in processing data and creating patterns for use in decision

making. Deep learning is a subset of machine learning in artificial intelligence that

has networks capable of learning unsupervised from data that is unstructured or

unlabeled. Also known as deep neural learning or deep neural network.[2]

Deep Learning models, with their multi-level structures are very helpful in

extracting complicated information from input images. Convolutional neural

networks are also able to drastically reduce computation time by taking advantage

of GPU for computation which many networks fail to utilize.

Fig 2.1 - Convolutional Neural Network

6



<a name="br11"></a> 

There are many sources of noise in images, and these noises come from various

aspects such as image acquisition, transmission, and compression. The types of

noise are also different, such as salt and pepper noise, Gaussian noise, etc. There

are different processing algorithms for different noises.

Fig 2.2 - Image, Noisy and Denoised

We see the importance of perceptually-motivated losses when the resulting image is

to be evaluated by a human observer. We compare the performance of several

losses, and utilize a variety of differentiable error functions for our model. We show

that the quality of the results improves significantly with better loss functions, even

when the network architecture is left unchanged.[1]

Fig 2.3 - Loss Function Variations

And finally, we deal with Data Augmentation. It is a strategy that enables

practitioners to significantly increase the diversity of data available for training

models, without actually collecting new data. Data augmentation techniques such as

cropping, padding, horizontal flipping are used to train large neural networks.[7]

Fig 2.4 - Data Augmentation Chart

7



<a name="br12"></a> 

**CHAPTER 3**

**PROJECT ANALYSIS/ PROJECT IMPLEMENTATION**

3\.1 Technology Used and Basic Framework

The basic working principle of the project is dependent upon Deep Learning algorithms

through which we analyze the noisy and corrupted image and run it through an optimized

model that we have trained to give the most accurate results in as little time as possible.

The implementation of the Deep Learning algorithm is done through Python mainly and

by making use of third-party libraries that are commonly used for Machine Learning

projects to make working much more easier. They are:

●

**NumPy:** It is a library for the Python programming language, adding support for

large, multidimensional arrays and matrices, along with a large collection of

high-level mathematical functions to operate on these arrays.

●

●

●

**Google Colab Patches:** Collaboration specific python libraries with a variety of

different functions that provide ease of use.

**TensorFlow:** It is a software library for data flow and differentiable programming

across a range of tasks.

**Multiprocessing:** It is a package that supports spawning processes using an API

similar to the threading module.

3\.2 Details of the Deep Learning Model

3\.2.1 Workflow

A simple flowchart of the standard methodology is given below. The provided flow

chart forms the basis of most Machine Learning operations though depending on

the conditions, a step may be minimized or expanded. For example, for a dataset

that has already been cleaned and provided to the researcher, the data cleaning and

transformation part of data extraction is thoroughly minimized. In our current

project, we follow each and every step in order to produce the output required by

the problem statement.[3]

**Dataset** - We have used the [Dogs](https://www.kaggle.com/c/dogs-vs-cats)[ ](https://www.kaggle.com/c/dogs-vs-cats)[vs](https://www.kaggle.com/c/dogs-vs-cats)[ ](https://www.kaggle.com/c/dogs-vs-cats)[Cats](https://www.kaggle.com/c/dogs-vs-cats)[ ](https://www.kaggle.com/c/dogs-vs-cats)dataset. It contains pictures of cats and

dogs. The reason to pick this dataset was that it was very not very convenient to

upload 600 MB of dataset repeatedly in google collab. Since we had its fast API,

we were able to train many models with the collab instances. Also, we wanted the

Neural Network model to understand the picture. The model should understand the

images and use its knowledge to fill the noise. (e.g. If an eye of the cat is 'noised'




<a name="br13"></a> 

than the model should know how an 'eye' looks like to fill in the noise) Since to do

the last part in the real world would require lots more data and computational

power, the above dataset was adequate to train a prototype.While doing our

training, we used 2000 images for training and another 1000 images for validation.

While feeding the images to the pipeline, it gets resized to (512,512). This size was

chosen because it would provide denoising on a reasonably decent size of an image

and was computationally less expensive.

Fig 3.1 - Planning Model

The process by which we have obtained these models is done through exposure to

different types of noises, data augmentation of the image database and finally,

giving it a self supervising nature. The details of which are as follows.

3\.2.2 Noise

In this project, we have considered three noises. The Neural Network will be

trained to take care of all these three noises. The noises taken into consideration are:

pepper, salt, localvar. [5]

**Salt-and-pepper noise** is a form of noise sometimes seen on images. It is also

known as impulse noise. This noise can be caused by sharp and sudden

disturbances in the image signal.

While **localvar** is zero-mean Gaussian white noise with an intensity-dependent

variance.



<a name="br14"></a> 

Before the image is fed to model for training. It would be assigned layers of any of

the above three noises randomly with equal probability.

Fig 3.2 - Noise Chart

3\.2.3 Data Augmentation

Data augmentation is used when we don’t have enough training data. Also, it has

the effect of regularization. In dealing with deep learning models, too much

learning is also bad for the model to predict with unseen data. If we get good results

in training data and poor results in unseen data (test data, validation data) then it is

framed as an overfitting problem. [2]

The usual way of doing the data augmentation is by flipping, cropping, flipping,

zooming, shearing etc. But here we have taken a different approach. Every time the

noise produced by the noise function differ, even if the noise-type and the image are

the same. This helps us to feed the model with unique images every time. Apart

from this, as discussed above each image fed to the model would be randomly

assigned a noise type in every epoch. Thus creating even more augmentation.



<a name="br15"></a> 

3\.2.4 Self-Supervising Nature

Given a task and enough labels, supervised learning can solve it well. Good

performance usually requires a decent amount of labels, but collecting manual

labels is expensive and hard to be scaled up. So to get examples of images which

are first corrupted and another image which is clean manually is expensive.

Therefore we are using self-supervised learning, where using a subset of the

information we are trying to predict the whole information. In our project, the

subset being the corrupted image in which some information has been destroyed by

corruption and the Deep Learning model are tasked to find the original image which

has full information.

In the context of an optimization algorithm, the function used to evaluate a

candidate solution (i.e. a set of weights) is referred to as the objective function. We

may seek to maximize or minimize the objective function, meaning that we are

searching for a candidate solution that has the highest or lowest score respectively.

Typically, with neural networks, we seek to minimize the error. As such, the

objective function is often referred to as a cost function or a loss function and the

value calculated by the loss function is referred to as simply “loss.” The cost or loss

function has an important job in that it must faithfully distill all aspects of the

model down into a single number in such a way that improvements in that number

are a sign of a better model. In this project, we have used five loss functions to

evaluate our models: **binary cross-entropy, mean absolute error, mean squared**

**error, mean squared logarithmic error, root mean square error**.[6]

3\.2.5 Evaluation Methodology

As stated earlier, we would be utilizing novel differentiable loss functions to further

optimize our models. Since our models would be training with different loss

functions, how would we decide which of this loss function is most suitable for that

model? Since each training will be optimized for its respective loss function, it is

hard to judge based on the loss function we are training. For that, we introduce two

metric called PSNR and SSIM. [4]

**PSNR** - Peak signal-to-noise ratio, often abbreviated PSNR, is an engineering term

for the ratio between the maximum possible power of a signal and the power of

corrupting noise that affects the fidelity of its representation. PSNR is most

commonly used to measure the quality of reconstruction of lossy compression

codecs (e.g., for image compression). When comparing compression codecs, PSNR

is an approximation to the human perception of reconstruction quality.

**SSIM** - structural similarity index measure is a method for predicting the perceived

quality of digital television and cinematic pictures, as well as other kinds of digital

images and videos. The difference with other techniques such as PSNR is that these

approaches estimate absolute errors. Structural information is the idea that the

pixels have strong inter-dependencies especially when they are spatially close.

These dependencies carry important information about the structure of the objects

in the visual scene.



<a name="br16"></a> 

**Visual Inspection** - We might have a good SSIM or PSNR it still does not convey

the complete information. For example, while doing the task, we came across

moments where one loss function would have a better result compared to another

loss function but it did not have the same visual appearance as it would be in

greyscale. But the other one was at least colourful. Therefore we would give more

preference to the latter one.

Note - In this project, the SSIM and PSNR are tweaked a little because they were

initially meant to be a loss function. Generally speaking the more the score for the

above two parameters the better, but to be used as a loss function, we have

subtracted the score from a constant. For SSIM it gets subtracted from the constant

1, for PSNR it's 100.

Utilizing all the above concepts, our end result provides us with two different

multi-layer model plans, the details of which are given as follows.

Fig 3.3 - Model 1(left) and 2(right)




<a name="br17"></a> 

**CHAPTER 4**

**RESULTS & DISCUSSIONS**

We have obtained multi-layer keras models that can produce our desired outputs

and have some slight variations over different loss functions. In total we trained 6

models 5 times each to get the perfect model.

Initially we implemented the models on a smaller scale to not overuse on functional

power of the device, however the results yielded were not quite optimal.This was

basic CNN without any fancy reduction or enhancement.

4\.1 Small Model 1

Table 4.1 - Loss Function Table, Small Model 1

Mean Absolute Error:

Noised Image

Original Image

Output Image

Mean Squared Error:




<a name="br18"></a> 

Noised Image

Original Image

Output Image

Mean Squared Logarithmic Error:

Noised Image

Original Image

Original Image

Original Image

Output Image

Output Image

Output Image

Binary Cross-entropy:

Noised Image

Root Mean Squared Error:

Noised Image




<a name="br19"></a> 

Table 4.2 Small Model 1 Results

4\.2 Small Model 2

Table 4.3 - Loss Function Table, Small Model 2

Mean Absolute Error:

Noised Image

Original Image

Output Image

Mean Squared Error:

Noised Image

Original Image

Output Image




<a name="br20"></a> 

Mean Squared Logarithmic Error:

Noised Image

Original Image

Original Image

Original Image

Output Image

Output Image

Output Image

Binary Cross-entropy:

Noised Image

Root Mean Squared Error:

Noised Image




<a name="br21"></a> 

Table 4.4 Small Model 2 Results

Looking at the results of the small scale models, we can see the output images are

not up to mark. At the expense of denoising, there is a great color desaturation. In

case of Model 1 with Binary Cross Entropy, the output image cannot even be

produced successfully. Therefore we can conclude that working on a small scale

model to reduce expense of processing power will not produce satisfactory results,

therefore we will run the models now on the intended scale and note the outputs.

4\.3 Normal Model 1

Table 4.5 - Loss Function Table, Regular Model 1

Mean Absolute Error:

Noised Image

Original Image

Output Image

Mean Squared Error:

Noised Image

Original Image

Output Image




<a name="br22"></a> 

Mean Squared Logarithmic Error:

Noised Image

Original Image

Original Image

Original Image

Output Image

Output Image

Output Image

Binary Cross-entropy:

Noised Image

Root Mean Squared Error:

Noised Image




<a name="br23"></a> 

Table 4.6 Model 1 Results

4\.4 Normal Model 2

Table 4.7 - Loss Function Table, Regular Model 2

Mean Absolute Error:

Noised Image

Original Image

Output Image

Mean Squared Error:

Noised Image

Original Image

Output Image




<a name="br24"></a> 

Mean Squared Logarithmic Error:

Noised Image

Original Image

Original Image

Original Image

Output Image

Output Image

Output Image

Binary Cross-entropy:

Noised Image

Root Mean Squared Error:

Noised Image




<a name="br25"></a> 

Table 4.8 Model 2 Results

The above models were a bit advanced compared to the small models. The Model 1

had skip connection for faster gradient descent. While model 2 was symmetric skip

autoencoder. After looking at the working of models on the intended scale, we can

conclude that our objective is being accomplished to some degree. Our model can

definitely recognize the subject of the image and reduce noise to a great degree,

however the issue of desaturation is not fully dealt with. We can make this

statement because the subject of the picture has little to no desaturation, however

the colors of the background are warped to some degree. And despite successful

denoising, there is a loss in clarity somewhat. Also we see the effect of auto

encoder blurring to some effect. Therefore, we have attempted to optimize the

model to see if even better results are yielded.

4\.5 Enhanced Model 1

Table 4.9 - Loss Function Table, Enhanced Model 1

Mean Absolute Error:

Noised Image

Original Image

Output Image




<a name="br26"></a> 

Mean Squared Error:

Noised Image

Original Image

Original Image

Original Image

Output Image

Output Image

Output Image

Mean Squared Logarithmic Error:

Noised Image

Binary Cross-entropy:

Noised Image




<a name="br27"></a> 

Root Mean Squared Error:

Noised Image

Original Image

Output Image

Table 4.10 Enhanced Model 1 Results

4\.6 Enhanced Model 2

Table 4.11 - Loss Function Table, Enhanced Model 2

Mean Absolute Error:

Noised Image

Original Image

Output Image




<a name="br28"></a> 

Mean Squared Error:

Noised Image

Original Image

Original Image

Original Image

Output Image

Output Image

Output Image

Mean Squared Logarithmic Error:

Noised Image

Binary Cross-entropy:

Noised Image





<a name="br29"></a> 

Root Mean Squared Error:

Noised Image

Original Image

Output Image

Table 4.12 Enhanced Model 2 Results

The model 2 was an autoencoder model which took an image, compressed it using

an encoder and then again expanded on it by a decoder. So we thought about using

an inverted form of autoencoder.(i.e. We first expand the image with the help of

encoder and then compress it to its right size). Looking at the results from the

enhanced models we can see that the issues of desaturation and loss of clarity have

been almost completely eradicated, leaving us with an almost fully accurate image

restoration. Thus our desired result was finally achieved. One interesting

observation we saw during this was that noise on subjects was removed much better

than the background noise. It shows that the neural network learned how a cat of

dogs looks and uses its knowledge to fill the noise on them much better.

Fig 4.1 - Showing the observation where the model denoised the portion of cat

nicely(Green box) but was not able to fill the noise of background(red box).





<a name="br30"></a> 

**CHAPTER 5**

**CONCLUSION AND FUTURE WORK**

5\.1 Conclusion

As we can see, utilizing multi-layer deep learning models and applying

differentiable loss functions as hyperparameters, we can effectively restore noised

images through some decent processing power. Initially we attempted to achieve

the same through small scale models, however it resulted in total desaturation of

output image. Hence we used the intended scale which gave us better results with

subject recognition and minimization of desaturation, however loss of clarity

remained an issue. Thus finally, we used fully optimized and enhanced models

which produced nearly fully accurate restorations.

5\.2 Future Work

Now that we have effectively produced self-supervising models with variations that

can adapt to multiple types of noise, our future scope would be to make our work

more easily approachable by providing its services through a mobile application or

online website. This would allow a layman with no knowledge of using python

systems to utilize the image de-noising service.

5\.

3 Planning And Project Management

**Activity**

**Starting week**

**Number of weeks**

Background Study and Research

Finalizing Plan/Approach

1<sup>st</sup> week of July

1<sup>st</sup> week of August

2<sup>nd</sup> week of August

4

1

2

Data Augmentation of Image

Database

Noising of Images

Designing the Model

Applying Loss Functions

1<sup>st</sup> week of September

2<sup>nd</sup> week of September

2<sup>nd</sup> week of October

1

3

2

26



<a name="br31"></a> 

Optimizing Model

1<sup>st</sup> week of November

Table 5.1 Schedule

2

**The Gantt chart is shown below:**

Fig 5.1 Gantt Chart

**REFERENCES**

[1] [*https://www.cambridgeincolour.com/tutorials/image-noise-2.htm*](https://www.cambridgeincolour.com/tutorials/image-noise-2.htm)

[2] [*https://cs.stanford.edu/people/rak248/VG_100K_2/*](https://numpy.org/)[* ](https://numpy.org/)v

[3] [*https://deepai.org/machine-learning-glossary-and-terms/*](https://deepai.org/machine-learning-glossary-and-terms/)

[4] [*https://matplotlib.org/*](https://matplotlib.org/)

[5] “Introduction to image de-noising” by Nabil Madali

[6] Image Restoration, Computer Vision

[7] Multi-Scale Structural Similarity Index for Image Quality

[8] Standford Engineering, CS, Image Processing and All You Need to Know, Prof.

Wang Lee, Prof. Elliot Junes





<a name="br32"></a> 

**INDIVIDUAL CONTRIBUTION REPORT:**

**IMAGE RESTORATION**

BISWAJEET SAHOO

1705689

**Abstract:** The commonly occurring issue of image restoration is what we are attempting

to solve through our project. We achieve this through deep learning models. But not only

are we attempting to restore the original image once it has been corrupted, we are also

training the model to be the most efficient model possible that can also adapt to new

types of noise and produce an accurate output image.

**Individual contribution and findings:** I worked on model creation along with my

fellow teammate Kush. In addition to that, I was mainly involved in training and testing

the models as well as evaluating them in an attempt to find out the best possible model

for our project. I was also involved in some visualization work.

**Individual contribution to project report preparation:** My contribution to project

report preparation was regarding model training, their results and its evaluation.

**Individual contribution for project presentation and demonstration:** During the

presentation I was responsible for the work in producing the slides related to my

individual contribution to the bulk of the project, as well as being responsible for any

explanation or demonstration related to them.

Full Signature of Supervisor:

…………………………….

Full signature of student:

Biswajeet Sahoo





<a name="br33"></a> 

**INDIVIDUAL CONTRIBUTION REPORT:**

**IMAGE RESTORATION**

DAIBIK DASGUPTA

1705692

**Abstract:** The commonly occurring issue of image restoration is what we are attempting

to solve through our project. We achieve this through deep learning models. But not only

are we attempting to restore the original image once it has been corrupted, we are also

training the model to be the most efficient model possible that can also adapt to new

types of noise and produce an accurate output image.

**Individual contribution and findings:** I have worked on the initial research of prior

attempts at image restoration model along with fellow teammate Kush. Through our

findings, we were able to develop the plan to produce a growing and adaptive model, as

well as the specifics about how to go about it, and how to make a model as efficient and

accurate as possible in this domain.

**Individual contribution to project report preparation:** I was involved in initial

research of producing a model that can achieve our end goal. I also worked on finding

suitable image databases that we can work with. Finally, I have aided in the writing of

basic concepts regarding the same.

**Individual contribution for project presentation and demonstration:** During the

presentation I was responsible for the work in producing the slides related with my

individual contribution to the bulk of the project, as well as being responsible for any

explanation or demonstration related to them.

Full Signature of Supervisor:

…………………………….

Full signature of student:

Daibik DasGupta





<a name="br34"></a> 

**INDIVIDUAL CONTRIBUTION REPORT:**

**IMAGE RESTORATION**

KUSH JAYANK PANDYA

1705701

**Abstract:** The commonly occurring issue of image restoration is what we are attempting

to solve through our project. We achieve this through deep learning models. But not only

are we attempting to restore the original image once it has been corrupted, we are also

training the model to be the most efficient model possible that can also adapt to new

types of noise and produce an accurate output image.

**Individual contribution and findings:** I have worked on the initial research of prior

attempts at image restoration model along with fellow teammate Daibik. With respect to

finding I was mainly involved in evaluation parameters and augmentation. I was also

involved in creating end to end pipelining for our model evaluation. Because of which we

were able to train multiple models. Also I was involved with my fellow team members

Biswajeet for model creation.

**Individual contribution to project report preparation:** My contribution to project

report preparation was with respect pipelining and evaluation and data augmentation.

**Individual contribution for project presentation and demonstration:** During the

presentation I was responsible for the work in producing the slides related with my

individual contribution to the bulk of the project, as well as being responsible for any

explanation or demonstration related to them.

Full Signature of Supervisor:

Full signature of student:

Kush Jayank Pandya

…………………………….





<a name="br35"></a> 




