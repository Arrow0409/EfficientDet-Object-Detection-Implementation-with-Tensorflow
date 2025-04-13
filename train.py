import os
import argparse
import tensorflow as tf
import numpy as np
from tqdm.autonotebook import tqdm
import shutil

# Import from your modules
from dataset import CocoDataset, Resizer, Normalizer, Augmenter, prepare_data_pipeline
from efficientdet import EfficientDet


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation in TensorFlow")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--test_interval", type=int, default=1)
    parser.add_argument("--es_min_delta", type=float, default=0.0)
    parser.add_argument("--es_patience", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="data/COCO")
    parser.add_argument("--log_path", type=str, default="tensorboard/tensorflow_efficientdet_coco")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    # Set random seed for reproducibility
    tf.random.set_seed(123)
    np.random.seed(123)
    
    # Setup training data pipeline
    train_dataset, train_coco_dataset, train_utils = prepare_data_pipeline(
        root_dir=opt.data_path,
        set_name="train2017",
        batch_size=opt.batch_size,
        common_size=opt.image_size,
        augmentation=True
    )
    
    # Setup validation data pipeline
    val_dataset, val_coco_dataset, val_utils = prepare_data_pipeline(
        root_dir=opt.data_path,
        set_name="val2017",
        batch_size=opt.batch_size,
        common_size=opt.image_size,
        augmentation=False
    )
    
    # Initialize model
    model = EfficientDet(num_classes=train_coco_dataset.num_classes())
    
    # Setup TensorBoard logging directory
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    
    # Setup model saving directory
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    
    # Create TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=opt.log_path,
        histogram_freq=1
    )
    
    # Setup optimizer with learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=opt.lr)
    
    # Create learning rate scheduler similar to PyTorch's ReduceLROnPlateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=3,
        verbose=1,
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    )
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=opt.es_min_delta,
        patience=opt.es_patience if opt.es_patience > 0 else float('inf'),
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    
    # Model checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(opt.saved_path, "tensorflow_efficientdet_coco.h5"),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    # Custom training loop
    best_loss = float('inf')
    best_epoch = 0
    
    # Train for specified number of epochs
    for epoch in range(opt.num_epochs):
        print(f"\nEpoch {epoch+1}/{opt.num_epochs}")
        
        # Training phase
        epoch_loss = []
        epoch_cls_loss = []
        epoch_reg_loss = []
        
        progress_bar = tqdm(train_dataset)
        for batch_idx, (images, annotations) in enumerate(progress_bar):
            with tf.GradientTape() as tape:
                # Forward pass
                cls_loss, reg_loss = model([images, annotations], training=True)
                
                # Calculate loss
                cls_loss = tf.reduce_mean(cls_loss)
                reg_loss = tf.reduce_mean(reg_loss)
                total_loss = cls_loss + reg_loss
            
            # Skip problematic batches
            if tf.math.is_nan(total_loss) or tf.math.is_inf(total_loss) or total_loss == 0:
                continue
            
            # Calculate gradients
            gradients = tape.gradient(total_loss, model.trainable_variables)
            
            # Clip gradients
            gradients, _ = tf.clip_by_global_norm(gradients, 0.1)
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Record losses
            epoch_loss.append(float(total_loss))
            epoch_cls_loss.append(float(cls_loss))
            epoch_reg_loss.append(float(reg_loss))
            
            # Update progress bar
            avg_loss = np.mean(epoch_loss)
            progress_bar.set_description(
                f'Epoch: {epoch+1}/{opt.num_epochs}. '
                f'Batch: {batch_idx+1}/{len(train_dataset)}. '
                f'Cls loss: {float(cls_loss):.5f}. '
                f'Reg loss: {float(reg_loss):.5f}. '
                f'Batch loss: {float(total_loss):.5f} '
                f'Avg loss: {avg_loss:.5f}'
            )
        
        # Log training metrics
        train_cls_loss = np.mean(epoch_cls_loss)
        train_reg_loss = np.mean(epoch_reg_loss)
        train_total_loss = np.mean(epoch_loss)
        
        print(f"Training - Classification Loss: {train_cls_loss:.5f}, "
              f"Regression Loss: {train_reg_loss:.5f}, "
              f"Total Loss: {train_total_loss:.5f}")
        
        # Update learning rate based on training loss
        reduce_lr.on_epoch_end(epoch, logs={'loss': train_total_loss})
        
        # Validation phase (every test_interval epochs)
        if epoch % opt.test_interval == 0:
            val_cls_losses = []
            val_reg_losses = []
            
            for val_images, val_annotations in val_dataset:
                # Forward pass in evaluation mode
                val_cls_loss, val_reg_loss = model([val_images, val_annotations], training=False)
                
                # Calculate mean losses
                val_cls_loss = tf.reduce_mean(val_cls_loss)
                val_reg_loss = tf.reduce_mean(val_reg_loss)
                
                val_cls_losses.append(float(val_cls_loss))
                val_reg_losses.append(float(val_reg_loss))
            
            # Calculate average validation losses
            val_cls_loss = np.mean(val_cls_losses)
            val_reg_loss = np.mean(val_reg_losses)
            val_total_loss = val_cls_loss + val_reg_loss
            
            print(f"Validation - Classification Loss: {val_cls_loss:.5f}, "
                  f"Regression Loss: {val_reg_loss:.5f}, "
                  f"Total Loss: {val_total_loss:.5f}")
            
            # Save model if validation loss improves
            if val_total_loss + opt.es_min_delta < best_loss:
                best_loss = val_total_loss
                best_epoch = epoch
                
                # Save the model
                model_path = os.path.join(opt.saved_path, "tensorflow_efficientdet_coco")
                model.save(model_path)
                print(f"Model saved to {model_path}")
                
                # Save model in TensorFlow SavedModel format
                tf.saved_model.save(model, os.path.join(opt.saved_path, "tensorflow_efficientdet_coco_saved_model"))
                
                # Convert to TensorFlow Lite format
                try:
                    converter = tf.lite.TFLiteConverter.from_saved_model(
                        os.path.join(opt.saved_path, "tensorflow_efficientdet_coco_saved_model")
                    )
                    tflite_model = converter.convert()
                    with open(os.path.join(opt.saved_path, "tensorflow_efficientdet_coco.tflite"), "wb") as f:
                        f.write(tflite_model)
                    print("TensorFlow Lite model saved successfully")
                except Exception as e:
                    print(f"Error converting to TFLite: {e}")
            
            # Check for early stopping
            if opt.es_patience > 0 and (epoch - best_epoch) > opt.es_patience:
                print(f"Early stopping triggered. Best loss: {best_loss:.5f} at epoch {best_epoch+1}")
                break
    
    print(f"Training completed. Best validation loss: {best_loss:.5f} at epoch {best_epoch+1}")


if __name__ == "__main__":
    opt = get_args()
    train(opt)