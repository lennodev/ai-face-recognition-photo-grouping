

# ai-face-recognition-photo-grouping
AI Face-recognition Photo Grouping project aims to group(find and copy) multiple users' photos into corresponding folders from large volume photo library. This project built by Python, Keras and Tensorflow. The Multi-task Cascaded Convolutional Networks (MTCNN) and FaceNet Model (developed by Hiroki Taniai) were adopted in this project. 

### Data Preparation
 - Register all target users via the system
 - Prepare some photo of those target users. The more sample photos you gave, the higher accuracy you will get!


### How it works:
 - Complete data preparation steps
 - Kick start the training NOW
 - Start grouping once training is completed
 - View corresponding users photo under the /output/<username>


### How to test fashion image classifier?

**User registration:**
```sh
{
    python main.py Register Peter #Adding Peter as registered user
}
```

**Start Training:**
```sh
{
    python main.py Train
}
```

**Start Grouping photo:**
```sh
{
    python main.py Group <Photo_library_path>
}
```


### Technology Used:
 - Python
 - Keras
 - Tensorflow

### ML Library/Model Used:
 - [Multi-task Cascaded Convolutional Networks (MTCNN)](https://github.com/ipazc/mtcnn)
 - [FaceNet by Hiroki Taniai](https://github.com/nyoki-mtl/keras-facenet)


### Training Dataset


### Folder Structure
``` bash
├── User										#folder to store user training photos
├── jupyter_notebook
│   ├── face-recognition-photo-grouping.ipynb	#jupyter notebook for better understanding
│   └── facenet_keras.h5						#FaceNet model hiroki taniai
├── model
│   ├── ModelLoader.py							#Model helper class
│   ├── embedding
│   │   └── embedding.npz						#generated training config
│   └── facenet_keras_hiroki_taniai.h5			#FaceNet model
├── service
│   ├── CommonService.py					
│   ├── DetectService.py
│   ├── FaceExtractService.py
│   ├── TrainService.py
├── util
│   ├── ConfigLoader.py
│   └── Logger.py
├── loader.py
├── main.py
├── output
└── requirements.txt
```

## Installation
It requires Python 3.7 and related dependenices

```sh
cd ai-face-recognition-photo-grouping/

#install related dependencies
pip install -r requirements.txt

```

### Todos
 - Reduce training speed by loading pre-processed UNKNOWN face vector

License
----

MIT
