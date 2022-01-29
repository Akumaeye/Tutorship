# import required packages
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
 
# Database paths : Training & testing images 
train_path = 'FER2013/train'
test_path = 'FER2013/test'

def DataLoading_and_preprocessing(train_path,test_path):
    
    # Data Loading
    # Emotions labels (Angry/Happy/Sad/Neutral/Surprise/Fear/Disgust)
    emotion_labels = sorted(os.listdir(train_path))
    # number of training classes (7)
    nb_classes = len(emotion_labels)
    
    # Preprocessing
    # Initialize image data generator with rescaling
    train_data_gen = ImageDataGenerator(rescale=1./255)
    validation_data_gen = ImageDataGenerator(rescale=1./255)
    
    # Preprocess all train images  
    train_generator = train_data_gen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
    
    # Preprocess all test images
    test_generator = validation_data_gen.flow_from_directory(
        test_path,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
    return train_generator, test_generator , nb_classes
    
print("[INFO] Data Loading & Data processing ..")    
train_generator, test_generator , nb_classes= DataLoading_and_preprocessing(train_path,test_path)  
  
def Emotion_model(nb_classes):
    
    # create model structure : CNN architecture
    emotion_model = tf.keras.Sequential()
    
    emotion_model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    
    emotion_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(tf.keras.layers.Dropout(0.25))
    
    emotion_model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    emotion_model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(tf.keras.layers.Dropout(0.25))
    
    emotion_model.add(tf.keras.layers.Flatten())
    
    emotion_model.add(tf.keras.layers.Dense(1024, activation='relu'))
    emotion_model.add(tf.keras.layers.Dropout(0.5))
    
    emotion_model.add(tf.keras.layers.Dense(nb_classes, activation='softmax'))
    
   
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    emotion_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(emotion_model.summary())
    
    return emotion_model

print("[INFO] Model creation ..")
emotion_model = Emotion_model(nb_classes)


def Training_model(train_generator, test_generator, emotion_model):
    
      # Train the neural network/model doe 60 epochs
      history = emotion_model.fit(x=train_generator,validation_data=test_generator,epochs=60)
      
      # Plotting loss and accuracy of validation set and training set 
      print(history.history.keys())
      plt.figure(1)
      
      # summarize history for accuracy
      plt.plot(history.history['accuracy'])
      plt.plot(history.history['val_accuracy'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
      
      # summarize history for loss
      plt.figure(2)
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
      
     # save trained model in .h5 file for predictions
      emotion_model.save('trained_models/emotion_model.h5')
    
print("[INFO] Training model ..")
Training_model(train_generator, test_generator, emotion_model)

# Database image plotting 
def plot_images(img_dir, top=10):
    all_img_dirs = os.listdir(img_dir)
    img_files = [os.path.join(img_dir, file) for file in all_img_dirs][:5]
  
    plt.figure(figsize=(10, 10))
  
    for idx, img_path in enumerate(img_files):
        plt.subplot(5, 5, idx+1)
    
        img = plt.imread(img_path)
        plt.tight_layout()         
        plt.imshow(img, cmap='gray') 
        
print("[INFO] Images Visualisation ..")        
plot_images(train_path+'/fear')   