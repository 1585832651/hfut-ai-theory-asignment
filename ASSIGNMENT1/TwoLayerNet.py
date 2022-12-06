import numpy as np
import matplotlib.pyplot as plt



class TwoLayerNet():
    def __init__(self,input_size,hidden_size,output_size,std=1e-4) -> None:
        self.params={}
        self.params["W1"]=std*np.random.randn(input_size,hidden_size)
        self.params["b1"]=np.zeros(hidden_size)
        self.params["W2"]=std*np.random.randn(hidden_size,output_size)
        self.params["b2"]=np.zeros(output_size)
        
        
    def loss(self,X,y,reg=0.0):
        W1,b1=self.params["W1"],self.params["b1"]
        W2,b2=self.params["W2"],self.params["b2"]
        N,D=X.shape
        scores=0
        
        input_2_layer1=X.dot(W1)+b1
        layer1_2_relu=np.max(0,input_2_layer1)
        layer1_2_layer2=layer1_2_relu.dot(W2)+b2  
        
        scores=layer1_2_layer2.dot(W2)+b2
        loss =0.0
        #softmax计算
        # scores = scores - np.max(scores,axis=1).reshape(-1,1)
        # exp_scores = np.exp(scores)
        # probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # correct_logprobs = -np.log(probs[range(N),y])
        # data_loss = np.sum(correct_logprobs) / N
        
        # reg_loss=reg*(np.sum(W1*W1)+np.sum(W2*W2))
        # loss=data_loss+reg_loss
        
        
        #RMSE计算
        num=X.shape[0]
        loss=np.sqrt(np.sum((y-scores)**2)/num)
        grads={}
        #RMSE关于y求导
        dscores = loss.copy()
        dscores=  -np.sum(y-(scores))   /np.sqrt( np.sum((y-scores)**2) /N)
        grads['W2'] = np.dot(layer1_2_layer2.T, dscores) + 2 * reg * W2
        grads['b2'] = np.sum(dscores, axis=0)
        grad_hidden_out = np.dot(dscores, W2.T)
        grad_hidden_in = (layer1_2_layer2 > 0) * grad_hidden_out
        grads['W1'] = np.dot(X.T, grad_hidden_in) + 2 * reg * W1
        grads['b1'] = np.sum(grad_hidden_in, axis=0)
        
        
        # dscores = loss.copy()
        # dscores[range(N), y] -= 1
        # # dscores /= N
        # # according to dimension analysis to calculate grads
        # grads['W2'] = np.dot(layer1_2_layer2.T, dscores) + 2 * reg * W2
        # grads['b2'] = np.sum(dscores, axis=0)
        # # do not forget the derivative of ReLU衍生物
        # grad_hidden_out = np.dot(dscores, W2.T)
        # grad_hidden_in = (layer1_2_layer2 > 0) * grad_hidden_out
        # grads['W1'] = np.dot(X.T, grad_hidden_in) + 2 * reg * W1
        # grads['b1'] = np.sum(grad_hidden_in, axis=0)
        
        return loss,grads
    
    def train(self,X,y,X_val,y_val,learning_rate=1e-3,
              learning_rate_decay=0.95,reg=5e-6,num_iters=100,
              batch_size=200,verbose=False):
        num_train=X.shape[0]
        iterations_per_epoch=max(num_train/batch_size,1)#这一没太大意义
        
        for it in range(num_iters):
            X_batch=0
            y_batch=0
            sample_index=np.random.choice(num_train,batch_size,replace=True)
            X_batch=X[sample_index]
            y_batch=y[sample_index]
            
            loss,grads=self.loss(X_batch,y_batch,reg=reg)
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']

        if verbose and it % 500 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        
        if it % iterations_per_epoch == 0:
                # Decay learning rate
                learning_rate *= learning_rate_decay
                
        print("train over")
        
    def predict(self,X):
        y_pred=None
        input_2_layer1=X.dot(self.params["W1"])+self.params["b1"]
        layer1_2_relu=np.maximum(0,input_2_layer1)
        score=layer1_2_relu.dot(self.params["W2"])+self.params["b2"]
        y_pred=np.argmax(score,axis=1)
        
        return y_pred
        