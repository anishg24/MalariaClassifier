# Malaria Classifier
This project is made by Anish Govind. Other projects can be found at my [GitHub](https://github.com/anishg24).
Huge thank you to the contributors behind this [dataset][dataset] as without it
this project wouldn't be published

![GitHub followers](https://img.shields.io/github/followers/anishg24?label=Follow&style=social)

![Status](https://img.shields.io/badge/status-completed-brightgreen?style=flat-square)
![Version](https://img.shields.io/github/v/release/anishg24/MalariaClassifier?color=orange&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/anishg24/MalariaClassifier?style=flat-square)

## Project Objective
Malaria is a very dangerous parasite that can be lethal. Around 2,0000 cases of malaria are diagnosed in the United States annually. 
This parasite is disastrous in poor regions like some parts in Africa. In 2018 alone, the WHO estimated that around 405,000 people died of malaria, and most
of which are in Africa. More information about this lethal parasite can be found [here](https://www.cdc.gov/malaria/about/faqs.html).

Images that should be used in the model should resemble:

<img src="https://storage.googleapis.com/kagglesdsdata/datasets/87153/200743/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_163.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584818866&Signature=CrIApf5op5knXyuh%2BTp469%2BwsZKaAHBRHh3sRXVozKW1sO39jWc3jCTbqTF%2B1YhuDA9q2jUBCsa3%2Bqx5ECBde3x%2F7z6H3cV%2FxAPJvHIRpW%2FOjBePiBXG1411Vdnh0x8hqtvU5NGPCuUzvjwBZKZtnawoNXaqXUlIDto3PhPW8jIhtcl0U7VlG1PbhPJNYzuEH2oT%2BQM6A73f7uAA12Id5b8va2b1HvCYxuXC6F%2FP0jE7dZPw50HUQ3O94ZmAwlhepnPUQGoGFnAJbQuN16%2Buxsmnh1n%2FmqRG0qjHIdl9qI6pZIROr4%2B8Ij9dXqhUHli5tRmVAhD8%2FwhYbxxQQ6Db%2Fg%3D%3D" width=100 align=right></img>
<img src="https://storage.googleapis.com/kagglesdsdata/datasets/87153/200743/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_131.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1584818896&Signature=WR1QQla6ZtSQKsVonbCd%2FXdz9AMGGTWzoRbTZY8eZ%2B%2BobRH0X6DNty6tc0nd4p8RNPR0IriaGeee3FR05qtffe7ArPTBwsdrYDLNXFxdVtv9rBuD0cl7cM8GzjE1TX0SZRRFu8mCTx%2B%2FoERtoGhRFdepHqkHIUG%2B%2Fv74VrMdsf%2F0avDNb1Vy4wdLQyTbfLV1qu66FeXIs3OAAaO0zCsP2yJQbrfm6wtSY1yJvV7t90av3N6lTWfpoG%2FV6%2BRPNYsLwSuj8eMh3KDwkLze8ygs529%2FTNsSW7Jt8gsSFqHqdvBFwAazhexBAZSgBWog8Es%2FwOibNpMPGBcpCqM5%2BuJRew%3D%3D" width=150 align=right></img>

Any other images **aren't** supported by this model.

### Methods Used
* Inferential Statistics
* Deep Learning
* Convolutional Neural Networks

### Technologies
* Python
* Pandas
* Jupyter & Matplotlib
* Numpy
* Sci-Kit Learn (for data handling)
* Open-CV (for resizing images)
* Keras & Tensorflow

## Project Log
This project was developed off the data (provided [here][dataset]) which consisted
of ~28,000 images of cells. The data is laid out so that there are 2 folders, labeled `Parasitized` and `Uninfected`.
Within those folders, are ~14,000 images of the cell respective to it's folder label. Armed with this, I thought I can
create a similar iterative sequence like my [Butterfly Classifier][bclass].

My initial thoughts were horribly horribly wrong. My other [classifiers][bclass] and my previous experience with the MNIST
dataset was all meant to fit within my computational limits. Sure training a model on a CPU is good for small models like
my [old one][bclass] but that had ~800 images. This [dataset][dataset] is ~35 times larger! When I created my first iteration,
I used a dictionary again (terrible idea) on my MacBook with 16GB of RAM. 

Initially, the dictionary was structured as such:
```python
my_dict = {
    "Infected": [1, 1, ...],
    "Uninfected": [0, 0, ...],
    "Image Files": [["C33P1thinF_IMG_20150619_114756a_cell_179.png", ...], ...], # List of just files, learning from my old mistakes
    "Image Arrays": [
        [
            [[[...]]], ... # 3-D Image matrices from matplotlib.image.imread. Resized to be 128x128.
        ],
        [
            [[[...]]], ... # 3-D Image matrices from matplotlib.image.imread. Resized to be 128x128.
        ],
        ...
    ]
}
```
And would create a DataFrame similar to the following:

Infected | Uninfected | Image Files | Image Arrays
------------ | ------------- | ------------ | ------------- |
1 | 0 | "C33P1thinF_IMG_20150619_114756a_cell_179.png" | [[[...]]]
1 | 0 | ... | ...
... | ... | ... | ...

After selecting my data that I care about (everything but Image Files) I saved my `.npy` files. This process was time consuming,
and took my 2013 MacBook about 20 minutes to complete. But nonetheless I completed the files and moved onto training.

Training was my first problem. CPU training is great for models that don't really exceed the 10,000 image mark
(though people who are really into ML would say that 2 days training is very small). It took my MacBook 50 minutes to complete 1 epoch out of 15.
This was unacceptable because one of the goals I had in mind was to make this model run light so that places in need of this model
can easily deploy it in a short time frame. (This goal led to a great sacrifice that will be dealt with later in this README).

I decided to research in training models with a GPU. All the current libraries that I use (Keras and tensorflow) only support
NVIDIA's architecture for programming, specifically CUDA programming. If you want to find out more visit [this](https://stackoverflow.com/questions/5211746/what-is-cuda-like-what-is-it-for-what-are-the-benefits-and-how-to-start).
Upon reading this I was intrigued. All my programming experience until now has been serial processing, but this new CUDA programming
introduces the concept of parallelization for me. Parallelization is essentially taking a job and splitting it across multiple cores (either CPU or GPU) to quickly 
accomplish the problem. CUDA is a parallelization option for NVIDIA GPU's. That's great but I use Apple devices to program on. Apple doesn't support NVIDIA GPUs anymore
since Mojave. So I decided to invest in a decked out [Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit).

Great now I have the computational power to make a model after setting up the OS and environment to support my program that too with its 128 Maxwell cores.
New problem. Jetson has 4 GB of un-expandable RAM. This is a problem because when my program processes data it saves the large arrays and everything to memory. I didn't have this issue with my MacBook
because that had 16 GB of RAM. The arrays that I created take ~5 GB of disk space. If I were to load the entire array into my memory in the `model.py` file, then I wouldn't have space to save my model
(which is also computed in memory). At this point I thought it would be impossible until I learned about streaming data.

Streaming solves the problem of processing larger-than-memory data. It takes a *batch* of data from a file, processes it, returns the result somewhere, then moves onto the next batch.
I found a library that is simple to use for streaming called [Dask](https://docs.dask.org/en/latest/). I have been tinkering with this library for ~5 weeks. Trying to learn it and apply it.
Sadly I have run out of time to work on this because I have picked up other projects.

I decided to instead trim the data to a point where I get a good accuracy with a computable amount of data. So I decided to take the first 2,500 images from each category and train my model.
I could compute this on my MacBook and get an accuracy of 70%. This means that places that need this model can quickly run this model to get results.

## Needs of this project

- Help on how to efficiently process the data (resizing an image to 128x128)

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Download the dataset [here][dataset] and unzip the file in the `data/` folder of this repo. 
   **DO NOT CHANGE THE FOLDER NAME FROM **`cell_images/`
3. `conda create venv`
4. `conda activate venv`
5. `conda install -r requirements.txt`
6. `python main.py ~/PATH/TO/YOUR/CELL_IMAGE`

Note: The `label_arrays.npy`, `image_arrays.npy`, and `butterfly_classifier.h5` files are **not** provided while cloning the repository.
This means that upon the first run, the script will automatically generate those files, but it will consume time and resources.
To get access to these files and quickly run the model, check out the [releases](https://github.com/anishg24/MalariaClassifier/releases). 
If you do decide to download the files and run the classifier, you don't need
to use the dataset, and can safely delete it.


## Arguments
Argument | Output
------------ | -------------
`~/PATH/TO/YOUR/IMAGE` | `INFECTED` or `NOT INFECTED`
`-h` `--help` | Show a help message and exit
`-r` `--retrain` | Retrains the model and runs the new model to predict off the given image. The new model gets saved in a file.
`-b` `--batch_size` | Changes the default batch size from 128 to user input
`-e` `--epochs` | Changes the default epoch size from 12 to user input
`-t` `--test_size` | Changes the default test proportion from 0.2 (20%) to user input.

## To-Do
- [x] Preprocess dataset
- [x] Create the model and save it
- [x] Make a working script to run everything and predict
- [x] Add the ability to retrain the model with new user input
- [ ] Use all 28k images to train the model (streaming)
- [ ] Data augmentation to improve accuracy of the models
- [ ] Make a website that allows you to add images or take images of a cell image and tells you if the cell is infected.

## Releases
- 1.0.0 (3/18/2020): First working release. No streaming yet, will come in v2.

## Contributing Members

Creator: [Anish Govind](https://github.com/anishg24@gmail.com)

Ways to contact:
* [E-Mail](anishg24@gmail.com)

**IF YOU FIND ANY ISSUES OR BUGS PLEASE OPEN AN ISSUE**

[dataset]: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
[bclass]: https://github.com/anishg24/ButterflyClassifier