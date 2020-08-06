#!/usr/bin/env python
# coding: utf-8

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)



############################
# neural network structure #
############################
tf.reset_default_graph()
# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])
# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 600], stddev=0.03, seed=12), name='W1')
b1 = tf.Variable(tf.random_normal([600], seed=12), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([600, 10], stddev=0.03, seed=12), name='W2')
b2 = tf.Variable(tf.random_normal([10], seed=12), name='b2')
# other variables
z1 = tf.get_variable('z1', W1.shape)
z2 = tf.get_variable('z2', b1.shape)
z3 = tf.get_variable('z3', W2.shape)
z4 = tf.get_variable('z4', b2.shape)
p = tf.get_variable('p', [1.])
lr = tf.placeholder(tf.float32, shape=[], name="lr")
beta = tf.placeholder(tf.float32, shape=[], name="beta")
keep_prob = tf.placeholder(tf.float32)
#neural net functions
hidden_out =  tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
error = tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1)
cross_entropy =  -tf.reduce_mean(error)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# input x - for 28 x 28 pixels = 784
x2 = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y2 = tf.placeholder(tf.float32, [None, 10])

W1_m2 = tf.Variable(tf.random_normal([784, 600], stddev=0.03, seed=12), name='W1_m2')
b1_m2 = tf.Variable(tf.random_normal([600], seed=12), name='b1_m2')
# and the weights connecting the hidden layer to the output layer
W2_m2 = tf.Variable(tf.random_normal([600, 10], stddev=0.03, seed=12), name='W2_m2')
b2_m2 = tf.Variable(tf.random_normal([10], seed=12), name='b2_m2')

N=mnist.train.images.shape[0]
dW1,db1,dW2,db2 = tf.gradients(xs=[W1,b1,W2,b2], ys=cross_entropy)
#update equations
W1_pri = tf.add(W1, p*lr*dW1+((beta-1.)/2*p+(beta+1)/2)*(2*lr)**0.5/N*z1, name='W1_pri')
b1_pri = tf.add(b1, p*lr*db1+((beta-1.)/2*p+(beta+1)/2)*(2*lr)**0.5/N*z2, name='b1_pri')
W2_pri = tf.add(W2, p*lr*dW2+((beta-1.)/2*p+(beta+1)/2)*(2*lr)**0.5/N*z3, name='W2_pri')
b2_pri = tf.add(b2, p*lr*db2+((beta-1.)/2*p+(beta+1)/2)*(2*lr)**0.5/N*z4, name='b2_pri')
W1_update = tf.assign(W1,W1_pri)
b1_update = tf.assign(b1,b1_pri)
W2_update = tf.assign(W2,W2_pri)
b2_update = tf.assign(b2,b2_pri)
z1_update = tf.assign(z1, tf.random_normal(W1.shape))
z2_update = tf.assign(z2, tf.random_normal(b1.shape))
z3_update = tf.assign(z3, tf.random_normal(W2.shape))
z4_update = tf.assign(z4, tf.random_normal(b2.shape))


W1_m2_ass = tf.assign(W1_m2, W1_pri)
b1_m2_ass = tf.assign(b1_m2, b1_pri)
W2_m2_ass = tf.assign(W2_m2, W2_pri)
b2_m2_ass = tf.assign(b2_m2, b2_pri)


with tf.control_dependencies([W1_m2_ass, b1_m2_ass, W2_m2_ass, b2_m2_ass]):
    #neural net functions
    hidden_out2 =  tf.nn.relu(tf.add(tf.matmul(x2, W1_m2), b1_m2))
    #hidden_out1 = tf.nn.dropout(hidden_out1, keep_prob) 
    y_2 = tf.nn.softmax(tf.add(tf.matmul(hidden_out2, W2_m2), b2_m2))
    y_clipped2 = tf.clip_by_value(y_2, 1e-10, 0.9999999)
    error2 = tf.reduce_sum(y2 * tf.log(y_clipped2) + (1 - y2) * tf.log(1 - y_clipped2), axis=1)
    cross_entropy2 =  -tf.reduce_mean(error2)
    correct_prediction2 = tf.equal(tf.argmax(y2, 1), tf.argmax(y_2, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
    

dW12,db12,dW22,db22 = tf.gradients(xs=[W1_m2,b1_m2,W2_m2,b2_m2], ys=cross_entropy2)

W1_update = tf.assign(W1,W1_pri)
b1_update = tf.assign(b1,b1_pri)
W2_update = tf.assign(W2,W2_pri)
b2_update = tf.assign(b2,b2_pri)

pf = .5
p_update = tf.assign(p, tf.to_float(tf.multinomial(tf.log([[pf, 1-pf]]),1)[0]*2-1))


def myflatten (val_list):
    return np.hstack([np.ravel(val) for val in val_list])

def mh_prob(flip, pf, beta, lr, dtheta_, dtheta_pri_, z_, N, cn, c, c_pri):
    beta1 = (beta-1)/2.*(-flip)+(beta+1)/2.
    beta2 = (beta-1)/2.*flip+(beta+1)/2.
    p1 = (pf-1/2.)*(-flip)+1/2.
    p2 = (pf-1/2.)*flip+1/2.
    temp1 = -1/(2*beta1**2)*np.linalg.norm(flip*N*(lr/2)**0.5*(dtheta_pri_-dtheta_)-beta2*z_)**2
    temp2 = -1/(2*beta2**2)*np.linalg.norm(-N*(lr/2)**0.5*(dtheta_pri_+dtheta_)-flip*beta2*z_)**2
    temp3 = -1/2.*np.linalg.norm(z_)**2
    temp4 = -1/(2*beta1**2)*np.linalg.norm(flip*N*(2*lr)**0.5*dtheta_+beta2*z_)**2
    max_temp = max([temp1,temp2,temp3,temp4])
    if (np.exp(temp3-max_temp)+np.exp(temp4-max_temp))==0:
        accept_prob=np.Inf
    else:     
        accept_prob = (p2*np.exp(temp1-max_temp-cn*(c_pri-c))+p1*np.exp(temp2-max_temp-cn*(c_pri-c)))/(p1*np.exp(temp3-max_temp)+p2*np.exp(temp4-max_temp))
    return min([accept_prob,1])

def set_beta (pf, beta_0, lr_, theta_0, N, cn, tol):
    #flip = sess.run(p_update)[0]
    p=-1
    flip = p
    ave_accept = 1000
    test_run = 100
    rep_num = 0
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([W1_update,b1_update,W2_update,b2_update], feed_dict={W1_pri:theta_0[0], b1_pri:theta_0[1], W2_pri:theta_0[2], b2_pri:theta_0[3]})
        while ave_accept>tol and rep_num<11 and ((1-rep_num*0.05)*beta_0)>1:
            beta_ = (1-rep_num*0.05)*beta_0
            ave_accept = 0.
            for i in range(test_run):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                batch_x2, batch_y2 = mnist.train.next_batch(batch_size=batch_size)
                z_val = sess.run([z1_update,z2_update,z3_update,z4_update])

                c, c_pri, dw1_v,db1_v,dw2_v,db2_v,dw12_v,db12_v,dw22_v,db22_v = sess.run([cross_entropy,cross_entropy2,dW1,db1,dW2,db2,dW12,db12,dW22,db22], feed_dict={x: batch_x, y: batch_y, beta:beta_, lr:lr_, x2:batch_x2, y2:batch_y2})

                dtheta_val = [dw1_v,db1_v,dw2_v,db2_v]
                dtheta_pri_val = [dw12_v,db12_v,dw22_v,db22_v]

                #compute acceptance probability
                dtheta_=myflatten(dtheta_val)
                dtheta_pri_=myflatten(dtheta_pri_val)
                z_=myflatten(z_val)
                accept_prob = mh_prob(flip, pf, beta_, lr_, dtheta_, dtheta_pri_, z_, N, cn, c, c_pri)
                ave_accept += accept_prob
                #print accept_prob
            ave_accept = ave_accept/test_run
            rep_num += 1
            print (beta_, ave_accept)
    return beta_, ave_accept

#######
# EXP #
#######

max_mc = []
max_sgd = []
max_sgld = []
times = 8
count = 0




for i in range(times):
    count += 1
    epochs=100
    batch_size=100
    cn=100
    lr_=0.4

    info_dir_path = "/home/leo/mcmc/result/mnist/exp1_batch/loop_{}/".format(count)


    total_batch = int(len(mnist.train.labels) / batch_size)
    #record results
    acc_test_mc=[]
    obj_test_mc=[]
    obj_train_mc=[]
    
    #dist_theta0_mc=[]
    iter_num=0;


    init_we_path = "/home/leo/mcmc/init_weight/500_1layer_init" + str(count) + "_gpu0.ckpt"

    saver = tf.train.Saver({"W1":W1, "b1":b1, "W2":W2, "b2":b2})

	#########
	# RSGLD #
	#########

    with tf.Session() as sess:
        #initialise the variables
        sess.run(tf.global_variables_initializer())
#         init_we_path = "~/dick/init_weight/1000_1layer_init" + str(i) + ".ckpt"
        saver.save(sess, os.path.expanduser(init_we_path))
        theta_0 = sess.run([W1,b1,W2,b2])
        beta_, accept_test = set_beta (pf, 1000.,lr_, theta_0, N, cn,0.7)
        print (beta_, accept_test)
        for epoch in range(epochs):
            avg_cost = 0
            avg_accept = 0
            accept_gd = []
            accept_ga = []
            iter_info = [] # (error, grad, accept prob, accept)

            for i in range(total_batch):
            	if i == 0:
                	batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                else:
                	batch_x = batch_x2
                	batch_y = batch_y2

                batch_x2, batch_y2 = mnist.train.next_batch(batch_size=batch_size)
                z_val = sess.run([z1_update,z2_update,z3_update,z4_update])
                flip = sess.run(p_update)[0]
                c, c_pri, dw1_v,db1_v,dw2_v,db2_v,dw12_v,db12_v,dw22_v,db22_v, W1_pri_val, b1_pri_val, W2_pri_val, b2_pri_val = sess.run([cross_entropy,cross_entropy2,dW1,db1,dW2,db2,dW12,db12,dW22,db22,W1_m2_ass,b1_m2_ass,W2_m2_ass, b2_m2_ass], feed_dict={x: batch_x, y: batch_y, beta:beta_, lr:lr_, x2:batch_x2, y2:batch_y2})

                dtheta_val = [dw1_v,db1_v,dw2_v,db2_v]
                dtheta_pri_val = [dw12_v,db12_v,dw22_v,db22_v]

#             compute acceptance probability
                dtheta_=myflatten(dtheta_val)
                dtheta_pri_=myflatten(dtheta_pri_val)
                z_=myflatten(z_val)
                accept_prob = mh_prob(flip, pf, beta_, lr_, dtheta_, dtheta_pri_, z_, N, cn, c, c_pri)
                if flip==-1:
                    accept_gd.append(accept_prob)
                else:
                    accept_ga.append(accept_prob)
                                                                                    
                u=np.random.rand(1)
                if u<accept_prob:
                    sess.run([W1_update,b1_update,W2_update,b2_update], feed_dict={W1_pri:W1_pri_val, b1_pri:b1_pri_val, W2_pri:W2_pri_val, b2_pri:b2_pri_val})
                    c = c_pri
                avg_accept += accept_prob
                avg_cost += c

                # save info

                # error
                acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

                # L2 gradient
                if u<accept_prob:
                	l2_grad = np.linalg.norm(np.array(dtheta_pri_).reshape(-1))
                else:
                	l2_grad = np.linalg.norm(np.array(dtheta_).reshape(-1))

                #
                iter_info.append([float(1-acc), float(l2_grad), float(accept_prob), float(u<accept_prob)])





            #record results at the end of each epoch    
            l = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
          
            a = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            
            avg_accept = avg_accept/total_batch
            avg_cost = avg_cost/total_batch

            obj_test_mc.append(l)
            obj_train_mc.append(c)
            acc_test_mc.append(a)
           	
           	# save info
            info_mc_path = info_dir_path + "rsgld/epoch{}_batch{}_lr{}_cn{}_loop{}.txt".format(epoch, batch_size, lr_, cn, int(count))

            if not os.path.exists(info_dir_path+'rsgld/'):
    			os.makedirs(info_dir_path+'rsgld/')

            np.savetxt(os.path.expanduser(info_mc_path), iter_info, fmt='%.4f')

            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "ave_accept =", "{:.3f}".format(avg_accept), "ave_accept_gd =", "{:.3f}".format(np.mean(accept_gd)), "ave_accept_ga =", "{:.3f}".format(np.mean(accept_ga)), "acc =", "{:.4f}".format(a))    
            #reset beta if prob too high
            if avg_accept>0.4:
                theta0 = sess.run([W1,b1,W2,b2])
                #batch_x0, batch_y0 = batch_x, batch_y
                beta_, accept_test = set_beta (pf, beta_,lr_, theta0, N, cn, 0.7)
                print (beta_, accept_test)
                #batch_x, batch_y = batch_x0, batch_y0
            if avg_accept<0.2:
                beta_=1.05*beta_
                print (beta_, avg_accept)

        print(max(acc_test_mc))
        max_mc.append(max(acc_test_mc))
        # file_name = "rsgld_epoch{}_batch{}_lr{}_cn{}_loop{}.txt".format(epochs,batch_size, lr_, cn, int(count))
        # acc_test_mc_path = "~/dick/result/mnist100/acc/rsgld_epoch{}_batch{}_lr{}_cn{}_loop{}.txt".format(epochs,batch_size, lr_, cn, int(count))
        # obj_test_mc_path = "~/dick/result/mnist100/cost_test/rsgld_epoch{}_batch{}_lr{}_cn{}_loop{}.txt".format(epochs,batch_size, lr_, cn, int(count))
        # obj_train_mc_path = "~/dick/result/mnist100/cost_train/rsgld_epoch{}_batch{}_lr{}_cn{}_loop{}.txt".format(epochs,batch_size, lr_, cn, int(count))
        # print(acc_test_mc_path)
        # np.savetxt(os.path.expanduser(acc_test_mc_path), acc_test_mc)
        # np.savetxt(os.path.expanduser(obj_test_mc_path), obj_test_mc)
        # np.savetxt(os.path.expanduser(obj_train_mc_path), obj_train_mc)
    
   
    ########
    # SGLD #
    ########

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_).minimize(cross_entropy)
    perturb_W1 = tf.add(W1, tf.random_normal(W1.shape, stddev=(2*lr)**0.5/N))
    perturb_b1 = tf.add(b1, tf.random_normal(b1.shape, stddev=(2*lr)**0.5/N))
    perturb_W2 = tf.add(W2, tf.random_normal(W2.shape, stddev=(2*lr)**0.5/N))
    perturb_b2 = tf.add(b2, tf.random_normal(b2.shape, stddev=(2*lr)**0.5/N))
   
    iter_num=0
    acc_test_sgld=[]
    obj_test_sgld=[]
    obj_train_sgld=[]
    cos_train_sgld=[]
    with tf.Session() as sess:
       # initialise the variables
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.expanduser(init_we_path))
        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            iter_info = []
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size, shuffle=True)  
                sess.run([optimizer], feed_dict={x: batch_x, y: batch_y})
                sess.run([perturb_W1, perturb_b1, perturb_W2, perturb_b2], feed_dict={lr:lr_})
                c, dw1_v, db1_v, dw2_v, db2_v = sess.run([cross_entropy, dW1, db1, dW2, db2], feed_dict={x: batch_x, y: batch_y})
                
                avg_cost += c

                # save info
                dtheta_val = [dw1_v,db1_v,dw2_v,db2_v]

                dtheta_=myflatten(dtheta_val)

                # error
                acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

                # L2 gradient
                
                l2_grad = np.linalg.norm(np.array(dtheta_).reshape(-1))
                

                iter_info.append([float(1-acc), float(l2_grad)])


            l = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            obj_test_sgld.append(l)
            obj_train_sgld.append(c)
            a = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            acc_test_sgld.append(a)
            avg_cost = avg_cost/total_batch

            info_mc_path = info_dir_path + "sgld/epoch{}_batch{}_lr{}_cn{}_loop{}.txt".format(epoch, batch_size, lr_, cn, int(count))

            if not os.path.exists(info_dir_path+'sgld/'):
    			os.makedirs(info_dir_path+'sgld/')

            np.savetxt(os.path.expanduser(info_mc_path), iter_info, fmt='%.4f')

            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "acc =", "{:.4f}".format(a), "lr =", "{:.2f}".format(lr_))

        # print(max(acc_test_sgld))
        # max_sgld.append(max(acc_test_sgld))

        # acc_test_sgld_path = "~/dick/result/mnist100/acc/sgld_epoch{}_batch{}_lr{}_cn{}_loop{}.txt".format(epochs,batch_size, lr_, cn, int(count))
        # obj_test_sgld_path = "~/dick/result/mnist100/cost_test/sgld_epoch{}_batch{}_lr{}_cn{}_loop{}.txt".format(epochs,batch_size, lr_, cn, int(count))
        # obj_train_sgld_path = "~/dick/result/mnist100/cost_train/sgld_epoch{}_batch{}_lr{}_cn{}_loop{}.txt".format(epochs,batch_size, lr_, cn, int(count))

        # print(acc_test_sgld_path)
        # np.savetxt(os.path.expanduser(acc_test_sgld_path), acc_test_sgld)
        # np.savetxt(os.path.expanduser(obj_test_sgld_path), obj_test_sgld)
        # np.savetxt(os.path.expanduser(obj_train_sgld_path), obj_train_sgld)
    # #######
    # # SGD #
    # #######

    # keep_prob_=1
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_).minimize(cross_entropy)
    # ########SGD##########
    # acc_test_sgd=[]
    # obj_test_sgd=[]
    # obj_train_sgd=[]
    # saver = tf.train.Saver({"W1":W1, "b1":b1, "W2":W2, "b2":b2})
    # with tf.Session() as sess:        
    #    # initialise the variables
    #     sess.run(tf.global_variables_initializer())
    #     saver.restore(sess, os.path.expanduser(init_we_path))

    #     total_batch = int(len(mnist.train.labels) / batch_size)

    #     for epoch in range(epochs):
    #         avg_cost = 0
    #         for i in range(total_batch):
    #             batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)            
    #             _,c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
    #             avg_cost += c 

    #         l = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

    #         obj_test_sgd.append(l)
    #         obj_train_sgd.append(c)
    #         a = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    #         acc_test_sgd.append(a)
    #         avg_cost = avg_cost/total_batch
    #         print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "acc =", "{:.4f}".format(a))
       
    #     print(max(acc_test_sgd))
    #     max_sgd.append(max(acc_test_sgd))

    #     acc_test_sgd_path = "~/dick/result/mnist100/acc/sgd_epoch500_batch100_lr0_01_loop" + str(count) + ".txt"
    #     obj_test_sgd_path = "~/dick/result/mnist100/cost_test/sgd_epoch500_batch100_lr0_01_loop" + str(count) + ".txt"
    #     obj_train_sgd_path = "~/dick/result/mnist100/cost_train/sgd_epoch500_batch100_lr0_01_loop" + str(count) + ".txt"
    #     print(acc_test_sgd_path)
    #     np.savetxt(os.path.expanduser(acc_test_sgd_path), acc_test_sgd)
    #     np.savetxt(os.path.expanduser(obj_test_sgd_path), obj_test_sgd)
    #     np.savetxt(os.path.expanduser(obj_train_sgd_path), obj_train_sgd)
    

# print("average acc mc = ", sum(max_mc)/len(max_mc))
# # print("average acc sgd = ", sum(max_sgd)/len(max_sgd))
# print("average acc sgld = ", sum(max_sgld)/len(max_sgld))
# print("max acc mc = ", max(max_mc))
# # print("max acc sgd = ", max(max_sgd))
# print("max acc sgld = ", max(max_sgld))

# acc_test_sgld_path = "~/dick/result/mnist100/acc/rsgld_epoch{}_batch{}_lr{}_allloop_acc.txt".format(epochs, batch_size, lr_)
# # obj_test_sgld_path = "~/dick/result/mnist100/acc/sgd_epoch500_batch100_lr0_01_allloop_acc.txt"
# obj_train_sgld_path = "~/dick/result/mnist100/acc/sgld_epoch{}_batch{}_lr{}_allloop_acc.txt".format(epochs, batch_size, lr_)
# np.savetxt(os.path.expanduser(acc_test_sgld_path), max_mc)
# # np.savetxt(os.path.expanduser(obj_test_sgld_path), max_sgd)
# np.savetxt(os.path.expanduser(obj_train_sgld_path), max_sgld)
        


