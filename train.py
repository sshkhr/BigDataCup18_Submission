# Training dataset
dataset_train = cracksDataset()

dataset_train.load_crack(subset="train")

dataset_train.prepare()

# Training dataset
dataset_val = cracksDataset()

dataset_val.load_crack(subset="val")

dataset_val.prepare()

import warnings
warnings.filterwarnings("ignore")

import imgaug

augmentation = imgaug.augmenters.Fliplr(.5)

#model.train(dataset_train, dataset_val, 
#            learning_rate=config.LEARNING_RATE, 
#           epochs=2, 
#          layers='heads',augmentation=augmentation)


#model.train(dataset_train, dataset_val, 
#            learning_rate=0.0001, 
#            epochs=2, 
#            layers='all',augmentation=augmentation)            

            
model.train(dataset_train, dataset_val, 
            learning_rate=0.001, 
            epochs=5, 
            layers='all',augmentation=augmentation)

model.keras_model.save_weights("mask_rcnn_shapes_aug_epoch_5.h5")            

model.train(dataset_train, dataset_val, 
            learning_rate=0.0001, 
            epochs=10, 
            layers='all',augmentation=augmentation)            

model.keras_model.save_weights("mask_rcnn_shapes_aug_epoch_10.h5")

model.train(dataset_train, dataset_val, 
            learning_rate=0.00001, 
            epochs=15, 
            layers='all',augmentation=augmentation)            

model.keras_model.save_weights("mask_rcnn_shapes_aug_epoch_15.h5")

model.train(dataset_train, dataset_val, 
            learning_rate=0.000001, 
            epochs=20, 
            layers='all',augmentation=augmentation)            

model.keras_model.save_weights("mask_rcnn_shapes_aug_epoch_20.h5")