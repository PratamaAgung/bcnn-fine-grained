from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from tflearn.data_utils import shuffle
import sklearn
import tensorflow as tf
import numpy as np
import h5py
import os
import pickle

if __name__ == "__main__":
    # with tf.device("/device:GPU:0"):
    resnet = ResNet50(include_top= False, weights='imagenet')

    feature = tf.placeholder(tf.float32, [None, 2, 2, 2048])
    target = tf.placeholder("float", [None, 10])

    #Define tensorflow graph
    sess = tf.Session()

    phi_I = tf.einsum('ijkm,ijkn->imn',feature,feature)
    print('Shape of phi_I after einsum', phi_I.get_shape())


    phi_I = tf.reshape(phi_I,[-1,2048*2048])
    print('Shape of phi_I after reshape', phi_I.get_shape())

    phi_I = tf.divide(phi_I,784.0)
    print('Shape of phi_I after division', phi_I.get_shape())

    y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))
    print('Shape of y_ssqrt', y_ssqrt.get_shape())

    z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)
    print('Shape of z_l2', z_l2.get_shape())

    fc3w = tf.get_variable('weights', [2048*2048, 10], initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    fc3b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32), name='biases', trainable=True)
    fc3l = tf.nn.bias_add(tf.matmul(z_l2, fc3w), fc3b)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3l, labels=target))
    learning_rate_wft = tf.placeholder(tf.float32, shape=[])
    learning_rate_woft = tf.placeholder(tf.float32, shape=[])

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.9, momentum=0.9).minimize(loss)


    correct_prediction = tf.equal(tf.argmax(fc3l,1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    y_pred = tf.argmax(fc3l, 1)

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    '''
    Load Training and Validation Data
    '''
    train_data = h5py.File('new_train.h5', 'r')
    val_data = h5py.File('new_val.h5', 'r')

    print('Input data read complete')

    X_train, Y_train = train_data['X'], train_data['Y']
    X_val, Y_val = val_data['X'], val_data['Y']
    print("Data shapes -- (train, val, test)", X_train.shape, X_val.shape)

    nb_epoch = 16
    batch_size = 2

    for epoch in range(nb_epoch):
        total_batch = int(len(X_train)/batch_size)
        X_train, Y_train = shuffle(X_train, Y_train)

        for i in range(total_batch):
            batch_xs, batch_ys = X_train[i*batch_size:i*batch_size+batch_size], Y_train[i*batch_size:i*batch_size+batch_size]
            pred = resnet.predict(batch_xs)

            sess.run(optimizer, feed_dict={feature: pred, target: batch_ys})

            cost = sess.run(loss, feed_dict={feature: pred, target: batch_ys})
            if i % 40 == 0:
                print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i,"Loss:", str(cost))
                print("Training Accuracy -->", sess.run(accuracy,feed_dict={feature: pred, target: batch_ys}))

        val_batch_size = 1
        total_val_count = len(X_val)
        correct_val_count = 0
        val_loss = 0.0
        total_val_batch = int(total_val_count/val_batch_size)
        for i in range(total_val_batch):
            batch_val_x, batch_val_y = X_val[i*val_batch_size:i*val_batch_size+val_batch_size], Y_val[i*val_batch_size:i*val_batch_size+val_batch_size]
            pred = resnet.predict(batch_val_x)

            val_loss += sess.run(loss, feed_dict={feature: pred, target: batch_val_y})

            pred = sess.run(num_correct_preds, feed_dict = {feature: pred, target: batch_val_y})
            correct_val_count+=pred

        print("##############################")
        print("Validation Loss -->", val_loss)
        print("correct_val_count, total_val_count", correct_val_count, total_val_count)
        print("Validation Data Accuracy -->", 100.0*correct_val_count/(1.0*total_val_count))
        print("##############################")

    class_prediction = []

    test_data = h5py.File('new_test.h5', 'r')
    X_test, Y_test = test_data['X'], test_data['Y']
    total_test_count = len(X_test)
    correct_test_count = 0
    test_batch_size = 1
    total_test_batch = int(total_test_count/test_batch_size)
    for i in range(total_test_batch):
        batch_test_x, batch_test_y = X_test[i*test_batch_size:i*test_batch_size+test_batch_size], Y_test[i*test_batch_size:i*test_batch_size+test_batch_size]
        pred = resnet.predict(batch_test_x)

        correct_pred, y_pred_batch = sess.run([num_correct_preds, y_pred], feed_dict = {feature: pred, target: batch_test_y})
        correct_test_count+= correct_pred
        class_prediction.extend(y_pred_batch)

    print("##############################")
    print("correct_test_count, total_test_count", correct_test_count, total_test_count)
    print("Test Data Accuracy -->", 100.0*correct_test_count/(1.0*total_test_count))
    print ("Precision ->", sklearn.metrics.precision_score(np.argmax(Y_test[:total_test_batch * test_batch_size], axis= 1), class_prediction, average='micro'))
    print ("Recall", sklearn.metrics.recall_score(np.argmax(Y_test[:total_test_batch * test_batch_size], axis= 1), class_prediction, average='micro'))
    print ("f1_score", sklearn.metrics.f1_score(np.argmax(Y_test[:total_test_batch * test_batch_size], axis= 1), class_prediction, average='micro'))
    print("##############################")
    print(class_prediction)

    saver = tf.train.Saver()
    saver.save(sess, os.getcwd() + "/bcnn_woft_residual")

    with open("class_pred_residual.pickle", "wb") as handle:
        pickle.dump(class_prediction, handle)
