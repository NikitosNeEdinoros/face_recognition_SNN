import os
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy  # type: ignore
from data_processing import prepare_data, data_partition, PATH_ANCH, PATH_NEG, PATH_POS
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, BatchNormalization  # type: ignore
from tensorflow.keras.losses import BinaryCrossentropy  # type: ignore

class L1SDist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(tf.convert_to_tensor(input_embedding) - tf.convert_to_tensor(validation_embedding))
        ###return tf.math.sqrt(tf.reduce_sum(tf.square(input_embedding - validation_embedding), axis=-1, keepdims=True)) ## Используем Евклидово расстояние

def create_embedding():
    input_img = Input(shape=(112, 112, 3), name='input')

    x = Conv2D(64, (10, 10), activation='relu')(input_img)
    x = MaxPooling2D(64, (2, 2), padding='same')(x)
    x = Conv2D(128, (7, 7), activation='relu')(x)
    x = MaxPooling2D(64, (2, 2), padding='same')(x)
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D(64, (2, 2), padding='same')(x)
    x = Conv2D(256, (4, 4), activation='relu')(x)
    x = Flatten()(x)
    output = Dense(4096, activation='sigmoid')(x)
    
    return Model(inputs=[input_img], outputs=[output], name='embedding')

def bring_together(embedding):
    input_image = Input(shape=(112, 112, 3), name='input_img')
    validation_image = Input(shape=(112, 112, 3), name='validation_img')

    dist_layer = L1SDist()
    distances = dist_layer(embedding(input_image), embedding(validation_image))

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='CNN_SNN')

@tf.function
def training_step(batch, model, loss_fun, optimizer):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        pred = model(X, training=True)
        loss = loss_fun(tf.reshape(y, (-1, 1)), pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, pred, y

def evaluate_model(model, test):
    accuracy = BinaryAccuracy()
    precision = Precision()
    recall = Recall()

    for batch in test:
        X = batch[:2]
        y_true = batch[2]
        pred = model(X, training=False)
        
        accuracy.update_state(y_true, pred)
        precision.update_state(y_true, pred)
        recall.update_state(y_true, pred)
    
    print(f"Test Accuracy: {accuracy.result().numpy():.4f}")
    print(f"Test Precision: {precision.result().numpy():.4f}")
    print(f"Test Recall: {recall.result().numpy():.4f}")

def train_proc(train, model, loss_fun, optimizer, num_epochs, checkpoint, checkpoint_pre):

    best_acc = 0

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        progbar = tf.keras.utils.Progbar(len(train))

        precision = Precision()
        recall = Recall()
        accuracy = BinaryAccuracy()
        epoch_loss = 0

        for index, batch in enumerate(train):
            loss, pred, y_true = training_step(batch, model, loss_fun, optimizer)
            epoch_loss += loss
            progbar.update(index + 1)

            precision.update_state(y_true, pred)
            recall.update_state(y_true, pred)
            accuracy.update_state(y_true, pred)

        avg_loss = epoch_loss / len(train)
        print(f" - loss: {avg_loss:.4f} - accuracy: {accuracy.result().numpy():.4f} - precision: {precision.result().numpy():.4f} - recall: {recall.result().numpy():.4f}")

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_pre)

if __name__ == '__main__':
    embedding = create_embedding()
    model = bring_together(embedding)
    model.summary()
    
    data = prepare_data()
    train, test = data_partition(data)
    
    loss_fun = BinaryCrossentropy()
    optimizer = tf.keras.optimizers.legacy.Adam(0.0001)

    checkpoint_dir = './train_checks'
    checkpoint_pre = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    train_proc(train, model, loss_fun, optimizer, 30, checkpoint, checkpoint_pre)
    model.save('./model/SNN_new.keras')

    evaluate_model(model, test)

