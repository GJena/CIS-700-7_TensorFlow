import tensorflow as tf


x_ = [[0, 0], [0, 1], [1, 0], [1, 1]] # input
expect=[[1,0],  [0,1],  [0,1], [1,0]] # one hot representation

x = tf.placeholder("float", [None,2])
y_ = tf.placeholder("float", [None, 2])

number_hidden_nodes = 30

def init_weights(shape, name):
    return tf.Variable(tf.random_uniform(shape), [-1.22, 1.22], name=name)

def model(X, w_h, w_o, b, b2):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("Layer2"):
        h = tf.nn.relu(tf.matmul(X, w_h)+b)
    with tf.name_scope("Layer3"):
        return tf.nn.softmax(tf.matmul(h, w_o)+b2)


#Initialize weights
w_h = init_weights([2, number_hidden_nodes], "w_h")
w_o = init_weights([number_hidden_nodes, 2], "w_o")
b = tf.Variable(tf.zeros([number_hidden_nodes]))
b2 = tf.Variable(tf.zeros([2]))

#Histogram summaries for weights
tf.histogram_summary("w_h_summ", w_h)
tf.histogram_summary("w_o_summ", w_o)

#Create Model (One hidden layer)
py_x = model(x, w_h, w_o, b ,b2)

#Cost function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_sum(y_*tf.log(py_x))
    train_op =  tf.train.GradientDescentOptimizer(0.2).minimize(cost)
    # Add scalar summary for cost tensor
    tf.scalar_summary("cost", cost)

#Measure accuracy
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(expect, 1), tf.argmax(py_x, 1)) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    # Add scalar summary for accuracy tensor
    tf.scalar_summary("accuracy", acc_op)

#Create a session
with tf.Session() as sess:
    writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph) #log writer
    merged = tf.merge_all_summaries()

    tf.initialize_all_variables().run()

    #Train the  model
    for i in range(1000):
        sess.run(train_op, feed_dict={x: x_, y_:expect })
        summary, acc, e = sess.run([merged, acc_op, cost], feed_dict={x: x_, y_:expect })
        writer.add_summary(summary, i)  # Write summary
        print(i, acc, e)                   # Report the accuracy
        if e<1:
            break

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(py_x,1), tf.argmax(y_,1)) # argmax along dim-1
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

    print "accuracy %s"%(accuracy.eval({x: x_, y_: expect}))

    learned_output=tf.argmax(py_x,1)
    print learned_output.eval({x: x_})                        
