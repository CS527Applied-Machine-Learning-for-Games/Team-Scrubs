from keras import backend as K
import tensorflow as tf
import SimpleITK as sitk
import numpy as np
# from itertools import izip


def dice_coef(y_true, y_pred):
    """Computes Dice coefficients with additive smoothing.

    :param y_true: one-hot tensor multiplied by label weights (batch size, number of pixels, number of labels).
    :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
    :return: Dice coefficients (batch size, number of labels).
    """
    smooth = 1.0
    y_true = K.cast(K.not_equal(y_true, 0), K.floatx())  # Change to binary
    intersection = K.sum(y_true * y_pred, axis=1)  # (batch size, number of labels)
    union = K.sum(y_true + y_pred, axis=1)  # (batch size, number of labels)
    return (2. * intersection + smooth) / (union + smooth)  # (batch size, number of labels)


def dice_loss(class_weights=[],nclass=2):
    def inner(y_true, y_pred):
        """Computes the average Dice coefficients as the loss function.

        :param y_true: one-hot tensor multiplied by label weights, (batch size, number of pixels, number of labels).
        :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
        :return: average Dice coefficient.
        """
        for i in range(nclass):
            cdice = 1.0-dice_coef(y_true[:,:,:,i], y_pred[:, :, :, i])
            loss=loss+cdice*class_weights[i]
        return loss
    return inner

def dice_loss_Im_mc(class_weights=[],nclass=2,ncluster=[]):
    """
    :param exp: exponent. 1.0 for no exponential effect, i.e. log Dice.
    """

    def inner(y_true, y_pred):
        """Computes the average exponential log Dice coefficients as the loss function.

        :param y_true: one-hot tensor multiplied by label weights, (batch size, number of pixels, number of labels).
        :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
        :return: average exponential log Dice coefficient.
        """
        mv=tf.reduce_max(y_true)
        I_weight = tf.cond(tf.greater(mv,50), lambda: tf.add(1.0,0.0), lambda: tf.add(0.0,0.0))
        S_weight = 1.0 - I_weight
        ind=nclass
        loss=0
        for i in range(nclass):
            yp=y_pred[:, :, :, i]
            for j in range(ncluster[i]-1):
                # w=np.zeros(L)
                # w[ind]=1
                yp=tf.add(yp,y_pred[:,:,:,ind])
                ind=ind+1
            cdice = 1.0-dice_coef(y_true[:,:,:,i], yp)
            loss=loss+cdice*class_weights[i]
        loss=loss*S_weight
        return loss

    return inner


def exp_dice_loss(exp=1.0, nclass=2,class_weights=[]):
    """
    :param exp: exponent. 1.0 for no exponential effect, i.e. log Dice.
    """

    def inner(y_true, y_pred):
        """Computes the average exponential log Dice coefficients as the loss function.

        :param y_true: one-hot tensor multiplied by label weights, (batch size, number of pixels, number of labels).
        :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
        :return: average exponential log Dice coefficient.
        """
        loss=0
        sumw=0
        for i in range(nclass):
            cdice = dice_coef(y_true[:,:,:,i], y_pred[:, :, :, i])
            cdice=K.clip(cdice, K.epsilon(), 1 - K.epsilon())  # As log is used
            cdice = K.pow(-K.log(cdice), exp)
            loss=loss+cdice*class_weights[i]
            sumw=sumw+class_weights[i]

        loss=loss/(nclass*sumw)
        # dice = dice_coef(y_true, y_pred) * class_weights
        # dice = K.clip(dice, K.epsilon(), 1 - K.epsilon())  # As log is used
        # dice = K.pow(-K.log(dice), exp)
        # if K.ndim(dice) == 2:
        #     dice = K.mean(dice, axis=-1)
        # return dice
        return loss
    return inner

def exp_dice_loss_Im_mc(exp=1.0, class_weights=[], nclass=2,ncluster=[]):
    """
    :param exp: exponent. 1.0 for no exponential effect, i.e. log Dice.
    """

    def inner(y_true, y_pred):
        """Computes the average exponential log Dice coefficients as the loss function.

        :param y_true: one-hot tensor multiplied by label weights, (batch size, number of pixels, number of labels).
        :param y_pred: softmax probabilities, same shape as y_true. Each probability serves as a partial volume.
        :return: average exponential log Dice coefficient.
        """
        mv=tf.reduce_max(y_true)
        I_weight = tf.cond(tf.greater(mv,50), lambda: tf.add(1.0,0.0), lambda: tf.add(0.0,0.0))
        S_weight = 1.0 - I_weight
        ind=nclass
        dice=0
        for i in range(nclass):
            yp=y_pred[:, :, :, i]
            for j in range(ncluster[i]-1):
                # w=np.zeros(L)
                # w[ind]=1
                yp=tf.add(yp,y_pred[:,:,:,ind])
                ind=ind+1
            cdice = dice_coef(y_true[:,:,:,i], yp)
            cdice = K.clip(cdice, K.epsilon(), 1 - K.epsilon())  # As log is used
            cdice = K.pow(-K.log(cdice), exp)
            dice=dice+cdice * class_weights[i]
        dice=dice/nclass*S_weight
        # dice = dice_coef(y_true, y_pred)
        # dice = K.clip(dice, K.epsilon(), 1 - K.epsilon())  # As log is used
        # dice = K.pow(-K.log(dice), exp)
        # if K.ndim(dice) == 2:
        #     dice = K.mean(dice, axis=-1)
        return dice

    return inner


def exp_categorical_crossentropy(exp=1.0, class_weights=[]):
    """
    :param exp: exponent. 1.0 for no exponential effect.
    """

    def inner(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # As log is used
        entropy = y_true * K.pow(-K.log(y_pred), exp)*class_weights
        entropy = K.sum(entropy, axis=-1)
        if K.ndim(entropy) == 2:
            entropy = K.mean(entropy, axis=-1)
        return entropy
    return inner

def exp_categorical_crossentropy_Im(exp=1.0, class_weights=[]):
    """
    :param exp: exponent. 1.0 for no exponential effect.
    """

    def inner(y_true, y_pred):
        mv=tf.reduce_max(y_true)
        I_weight = tf.cond(tf.greater(mv,50), lambda: tf.add(0.5,0.5), lambda: tf.add(0.0,0.0))
        S_weight = 1.0 - I_weight

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # As log is used
        entropy = y_true * K.pow(-K.log(y_pred), exp)*class_weights
        entropy = K.sum(entropy, axis=-1)
        if K.ndim(entropy) == 2:
            entropy = K.mean(entropy, axis=-1)
        return entropy * S_weight

        # if len(np.unique(y_true))>50:
        #     return 0
        # else:
        #     y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # As log is used
        #     entropy = y_true * K.pow(-K.log(y_pred), exp)*class_weights
        #     entropy = K.sum(entropy, axis=-1)
        #     if K.ndim(entropy) == 2:
        #         entropy = K.mean(entropy, axis=-1)
        #     return entropy
    return inner

def exp_categorical_crossentropy_Im_mc(exp=1.0, class_weights=[],nclass=2,ncluster=[]):
    """
    :param exp: exponent. 1.0 for no exponential effect.
    """

    def inner(y_true, y_pred):
        # sz=y_pred.shape
        mv=tf.reduce_max(y_true)
        I_weight = tf.cond(tf.greater(mv,50), lambda: tf.add(1.0,0.0), lambda: tf.add(0.0,0.0))
        S_weight = 1.0 - I_weight

        inds=np.zeros(nclass, np.int32)
        ind=nclass

        # L=np.sum(ncluster)
        # if S_weight<0:
        #     print(LL)
        # else:
        #     print(LLL)
        entropy = 0.0

        for i in range(nclass):
            inds[i] = ind
            for j in range(ncluster[i]-1):
                ind = ind+1
        '''
        yp0 = y_pred[:, :, :, 0]
        ind = inds[0]
        for j in range(ncluster[0] - 1):
            # if(i ==3 and y_true[:,:,1])
            # w=np.zeros(L)
            # w[ind]=1
            yp0 = tf.add(yp0, y_pred[:, :, :, ind])
            ind = ind + 1

        for i in [1,2,3,4]:
            yp0 = tf.add(yp0, y_true[:,:,:,0]*y_pred[:,:,:,i])

        yp0 = K.clip(yp0, K.epsilon(), 1 - K.epsilon())  # As log is used
        entropy = tf.add(entropy, y_true[:, :, :, 0] * K.pow(-K.log(yp0), exp) * class_weights[i])
        '''

        for i in [1,2,3,4]:
        #for i in [0,1]:
            yp = y_pred[:, :, :, i]
            ind = inds[i]
            for j in range(ncluster[i]-1):
                #if(i ==3 and y_true[:,:,1])
                # w=np.zeros(L)
                # w[ind]=1
                yp = tf.add(yp, y_pred[:,:,:,ind])
                ind = ind+1

            background = y_true[:, :, :, 0]
            yp = tf.subtract(yp, background)

            yp = K.clip(yp, K.epsilon(), 1 - K.epsilon())  # As log is used
            entropy=tf.add(entropy,y_true[:,:,:,i] * K.pow(-K.log(yp), exp) * class_weights[i])
        '''
        yp1 = y_pred[:, :, :, 1]
        ind = inds[1]
        
        for j in range(ncluster[1] - 1):
            yp1 = tf.add(yp1, y_pred[:, :, :, ind])
            ind = ind + 1
            
            
        yp4 = y_pred[:, :, :, 4]
        ind = inds[4]
        
        for j in range(ncluster[4] - 1):
            yp4 = tf.add(yp4, y_pred[:, :, :, ind])
            ind = ind + 1

        mv = tf.reduce_max(y_true[:,:,:,4])

        assignWeight = tf.cond(tf.greater(1.0, mv), lambda: tf.add(1.0, 0.0), lambda: tf.add(0.0, 0.0))

        yp1 = tf.add(yp1, yp4*assignWeight)
        yp1 = K.clip(yp1, K.epsilon(), 1 - K.epsilon())

        entropy = tf.add(entropy, y_true[:, :, :, 1] * K.pow(-K.log(yp1), exp) * class_weights[1])
        '''

        '''
        ############BEGIN PREVIOUS
        
        for i in range(nclass):
            yp=y_pred[:, :, :, i]
            for j in range(ncluster[i]-1):
                # w=np.zeros(L)
                # w[ind]=1
                yp=tf.add(yp,y_pred[:,:,:,ind])
                ind=ind+1

            yp = K.clip(yp, K.epsilon(), 1 - K.epsilon())  # As log is used
            entropy=tf.add(entropy,y_true[:,:,:,i] * K.pow(-K.log(yp), exp) * class_weights[i])

        '''
        # y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # As log is used
        # entropy = y_true * K.pow(-K.log(y_pred), exp) * class_weights
        # entropy = K.sum(entropy, axis=-1)
        if K.ndim(entropy) == 2:
            entropy = K.mean(entropy, axis=-1)
        return entropy * S_weight

        # if len(np.unique(y_true))>50:
        #     return 0
        # else:
        #     y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # As log is used
        #     entropy = y_true * K.pow(-K.log(y_pred), exp)*class_weights
        #     entropy = K.sum(entropy, axis=-1)
        #     if K.ndim(entropy) == 2:
        #         entropy = K.mean(entropy, axis=-1)
        #     return entropy
    return inner


def correlation_crossentropy_2D(NC=2):
    """
    :param exp: exponent. 1.0 for no exponential effect.
    """

    def inner(y_true, y_pred):
        # y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # As log is used
        # entropy = y_true * K.pow(-K.log(y_pred), exp)
        # entropy = K.sum(entropy, axis=-1)
        # if K.ndim(entropy) == 2:
        #     entropy = K.mean(entropy, axis=-1)
        sz = K.shape(y_pred)
        # sz=sz.num_elements()
        # print(sz)
        # if sz[1]!=262144:
        #     print(L2)
        #
        # if sz[2]!=2:
        #     print(L3)
        #
        # if sz[0]!=5:
        #     print(L1)


        # print(sz[0], sz[1], sz[2], sz[3])
        # y_true_1 = y_true[:,:,0:511] - y_true[:,:,1:512]
        y_true_1 = y_true[:, 0:sz[1] - 1, :,:] - y_true[:, 1:sz[1], :,:]
        # y_true_2 = y_true[:, 2:sz[1], :, :] - y_true[:, 1:sz[1]-1, :, :]
        y_true_3 = y_true[:, :, 0:sz[2] - 1,:] - y_true[:, :, 1:sz[2],:]
        # y_true_4 = y_true[:, :, 2:sz[2], :] - y_true[:, :, 1:sz[2]-1, :]

        y_pred_1 = y_pred[:, 0:sz[1] - 1, :,:] - y_pred[:, 1:sz[1], :,:]
        # y_pred_2 = y_pred[:, 2:sz[1], :, :] - y_pred[:, 1:sz[1]-1, :, :]
        y_pred_3 = y_pred[:, :, 0:sz[2] - 1,:] - y_pred[:, :, 1:sz[2],:]
        # y_pred_4 = y_pred[:, :, 2:sz[2], :] - y_pred[:, :, 1:sz[2]-1, :]

        corr_cost = K.mean(K.pow(y_true_1-y_pred_1,2)) + K.mean(K.pow(y_true_3-y_pred_3,2))
        cost1=0
        n=0
        for i in range(NC):
            for j in range(i):
                y_true_0 = y_true[:, :, :, i] - y_true[:, :, :, j]
                y_pred_0 = y_pred[:, :, :, i] - y_pred[:, :, :, j]
                cost1+=K.mean(K.pow(y_true_0-y_pred_0,2))
                # n++
        # cost1=cost1/n

        # y_pred[:,]
        return (corr_cost+cost1)

    return inner

def VoI_M_2D():
    """
    :param exp: exponent. 1.0 for no exponential effect.
    """

    def inner(y_true, y_pred):

        sz = K.shape(y_pred)

        M_true_00 = tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 2, 0:sz[2]-2,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_true_01 = tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 2, 1:sz[2]-1,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_true_02 = tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 2, 2:sz[2],:]  , y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_true_10 = tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1] - 1, 0:sz[2]-2,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_true_11 = tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1] - 1, 1:sz[2]-1,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_true_12 = tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1] - 1, 2:sz[2],:]  , y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_true_20 = tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1]    , 0:sz[2]-2,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_true_21 = tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1]    , 1:sz[2]-1,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_true_22 = tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1]    , 2:sz[2]  ,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)


        M_pred_00 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_pred_01 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_pred_02 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_pred_10 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_pred_11 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_pred_12 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_pred_20 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_pred_21 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
        M_pred_22 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 2:sz[2]  ,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)


        # M_true_00 = tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 1, 0:sz[2] - 1,:], y_true[:, 1:sz[1], 1:sz[2], :]), axis=-1)
        # M_true_01 = tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 1, :,:], y_true[:, 1:sz[1], :,:]), axis=-1)
        # M_true_02 = tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 1, 1:sz[2],:], y_true[:, 1:sz[1], 0:sz[2] - 1, :]), axis=-1)
        # M_true_10 = tf.reduce_sum(tf.minimum(y_true[:, :, 0:sz[2] - 1,:], y_true[:, :, 1:sz[2],:]), axis=-1)
        # M_true_11 = tf.reduce_sum(tf.minimum(y_true[:, :, :,:], y_true[:, :, :,:]), axis=-1)
        # M_true_12 = tf.reduce_sum(tf.minimum(y_true[:, :, 1:sz[2],:], y_true[:, :, 0:sz[2]-1, :]), axis=-1)
        # M_true_20 = tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1], 0:sz[2] - 1,:], y_true[:, 0:sz[1]-1, 1:sz[2],:]), axis=-1)
        # M_true_21 = tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1], :,:], y_true[:, 0:sz[1]-1, :,:]), axis=-1)
        # M_true_22 = tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1], 1:sz[2],:], y_true[:, 0:sz[1]-1, 0:sz[2]-1,:]), axis=-1)
        #
        # M_pred_00 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 1, 0:sz[2] - 1,:], y_true[:, 1:sz[1], 1:sz[2], :]), axis=-1)
        # M_pred_01 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 1, :,:], y_true[:, 1:sz[1], :,:]), axis=-1)
        # M_pred_02 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 1, 1:sz[2],:], y_true[:, 1:sz[1], 0:sz[2] - 1, :]), axis=-1)
        # M_pred_10 = tf.reduce_sum(tf.minimum(y_pred[:, :, 0:sz[2] - 1,:], y_true[:, :, 1:sz[2],:]), axis=-1)
        # M_pred_11 = tf.reduce_sum(tf.minimum(y_pred[:, :, :,:], y_true[:, :, :,:]), axis=-1)
        # M_pred_12 = tf.reduce_sum(tf.minimum(y_pred[:, :, 1:sz[2],:], y_true[:, :, 0:sz[2]-1,:]), axis=-1)
        # M_pred_20 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1], 0:sz[2] - 1,:], y_true[:, 0:sz[1]-1, 1:sz[2],:]), axis=-1)
        # M_pred_21 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1], :,:], y_true[:, 0:sz[1]-1, :,:]), axis=-1)
        # M_pred_22 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1], 1:sz[2],:], y_true[:, 0:sz[1]-1, 0:sz[2]-1,:]), axis=-1)

        M_true = M_true_00 + M_true_01 + M_true_02 + M_true_10 + M_true_11 + M_true_12 + M_true_20 + M_true_21 + M_true_22
        M_pred = M_pred_00 + M_pred_01 + M_pred_02 + M_pred_10 + M_pred_11 + M_pred_12 + M_pred_20 + M_pred_21 + M_pred_22
        M_join = tf.multiply(M_pred_00, M_true_00) + tf.multiply(M_pred_01, M_true_01) + tf.multiply(M_pred_02, M_true_02) + tf.multiply(M_pred_10, M_true_10) + tf.multiply(M_pred_11, M_true_11) + tf.multiply(M_pred_12, M_true_12) + tf.multiply(M_pred_20, M_true_20) + tf.multiply(M_pred_21, M_true_21) + tf.multiply(M_pred_22, M_true_22)


        en_true = - tf.reduce_mean(tf.log(M_true))
        en_pred = - tf.reduce_mean(tf.log(M_pred))
        en_join = - tf.reduce_mean(tf.log(M_join))

        VoI = en_join * 2 - en_pred - en_true
        # return tf.reduce_mean(tf.reduce_sum(y_true,axis=-1))
        return VoI
    return inner

def EncodingLength_2D(image_weight=1.0, L=1):
    """
    :param exp: exponent. 1.0 for no exponential effect.
    """
    def inner(y_true, y_pred):
        sz = K.shape(y_pred)
        mv=tf.reduce_max(y_true)
        I_weight = tf.cond(tf.greater(mv,50), lambda: tf.add(0.5,0.5), lambda: tf.add(0.0,0.0))
        S_weight = 1.0 - I_weight
        I_weight=I_weight*image_weight

        bins=256
        # sz = K.shape(y_pred)
        entropy_sum=0
        # entropy_seg=0
        im = tf.cast(tf.reshape(y_true[:, :, :, 0], [-1]),tf.int32)
        # seg_hist=tf.zeros(L,1)
        imsize=tf.reduce_sum(y_pred)

        for l in range(L):
            prob=tf.reshape(y_pred[:,:,:,l],[-1])
            # segsize=tf.reduce_sum(prob)+1E-10
            # entropy_seg =entropy_seg-segsize/imsize*tf.log(segsize/imsize)
            # meanprob=tf.reduce_mean(prob)
            # sumprob=tf.reduce_sum(prob)
            hist = tf.unsorted_segment_sum(prob, im, bins)
            logp=-tf.log(tf.divide(hist+1E-10,tf.reduce_sum(hist)+1E-5))
            entropy_sum=entropy_sum+tf.reduce_sum(tf.multiply(hist,logp))/imsize
        # seg_prob=tf.divide(seg_hist+1E-10,tf.reduce_sum(seg_hist))
        # entropy_seg = tf.reduce_sum(tf.multiply(seg_hist,-tf.log(seg_prob)))/tf.reduce_sum(seg_hist)

        return entropy_sum * I_weight
    return inner


def EncodingLength_mc_2D(image_weight=1.0, nclass=1, ncluster=[1]):
    """
    :param exp: exponent. 1.0 for no exponential effect.
    """
    def inner(y_true, y_pred):
        sz = K.shape(y_pred)
        mv=tf.reduce_max(y_true)
        I_weight = tf.cond(tf.greater(mv,50), lambda: tf.add(0.5,0.5), lambda: tf.add(0.0,0.0))
        S_weight = 1.0 - I_weight
        I_weight=I_weight*image_weight

        bins=256
        # sz = K.shape(y_pred)
        entropy_sum=0
        entropy_seg=0
        im = tf.cast(tf.reshape(y_true[:, :, :, 0], [-1]),tf.int32)
        # seg_hist=tf.zeros(L,1)
        imsize=tf.reduce_sum(y_pred)
        ind = nclass
        for i in range(nclass):
            prob=tf.reshape(y_pred[:,:,:,i],[-1])
            segsize=tf.reduce_sum(prob)+1E-10
            asegsize=segsize
            entropy_seg0=-segsize*tf.log(segsize)
            hist = tf.unsorted_segment_sum(prob, im, bins)
            logp=-tf.log(tf.divide(hist+1E-10,tf.reduce_sum(hist)))
            entropy_sum=entropy_sum+tf.reduce_sum(tf.multiply(hist,logp))/imsize
            for j in range(ncluster[i] - 1):
                # seg[seg == ind] = i
                prob = tf.reshape(y_pred[:, :, :, ind], [-1])
                ind = ind + 1
                hist = tf.unsorted_segment_sum(prob, im, bins)
                logp = -tf.log(tf.divide(hist + 1E-10, tf.reduce_sum(hist)))
                entropy_sum = entropy_sum + tf.reduce_sum(tf.multiply(hist, logp)) / imsize
                segsize=tf.reduce_sum(prob)+1E-10
                entropy_seg0 = entropy_seg0 - segsize * tf.log(segsize)
                asegsize=asegsize+segsize

            entropy_seg0=entropy_seg0+asegsize*tf.log(asegsize)
            entropy_seg = entropy_seg + entropy_seg0/asegsize

        return (entropy_sum+entropy_seg) * I_weight
    return inner

def VoI_Seg_mc_2D(y_true, y_pred, nclass=2, ncluster=[]):
    mv = tf.reduce_max(y_true)
    I_weight = tf.cond(tf.greater(mv, 50), lambda: tf.add(0.5, 0.5), lambda: tf.add(0.0, 0.0))
    S_weight = 1.0 - I_weight
    y_true=y_true * S_weight

    sz = K.shape(y_pred)
    M_true_00 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 2, 0:sz[2]-2,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_01 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 2, 1:sz[2]-1,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_02 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 2, 2:sz[2],:]  , y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_10 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1] - 1, 0:sz[2]-2,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_11 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1] - 1, 1:sz[2]-1,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_12 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1] - 1, 2:sz[2],:]  , y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_20 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1]    , 0:sz[2]-2,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_21 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1]    , 1:sz[2]-1,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_22 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1]    , 2:sz[2]  ,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)

    M_pred_00 = tf.multiply(M_true_00,0.0)
    M_pred_01 = tf.multiply(M_true_01,0.0)
    M_pred_02 = tf.multiply(M_true_02,0.0)
    M_pred_10 = tf.multiply(M_true_10,0.0)
    M_pred_11 = tf.multiply(M_true_11,0.0)
    M_pred_12 = tf.multiply(M_true_12,0.0)
    M_pred_20 = tf.multiply(M_true_20,0.0)
    M_pred_21 = tf.multiply(M_true_21,0.0)
    M_pred_22 = tf.multiply(M_true_22,0.0)

    ind = nclass
    for i in range(nclass):
        yp = y_pred[:, :, :, i]
        for j in range(ncluster[i] - 1):
            # w=np.zeros(L)
            # w[ind]=1
            yp = tf.add(yp, y_pred[:, :, :, ind])
            ind = ind + 1
        M_pred_00 = tf.add(M_pred_00,tf.minimum(yp[:, 0:sz[1] - 2, 0:sz[2] - 2], yp[:, 1:sz[1] - 1, 1:sz[2] - 1]))
        M_pred_01 = tf.add(M_pred_01,tf.minimum(yp[:, 0:sz[1] - 2, 1:sz[2] - 1], yp[:, 1:sz[1] - 1, 1:sz[2] - 1]))
        M_pred_02 = tf.add(M_pred_02,tf.minimum(yp[:, 0:sz[1] - 2, 2:sz[2]], yp[:, 1:sz[1] - 1, 1:sz[2] - 1]))
        M_pred_10 = tf.add(M_pred_10,tf.minimum(yp[:, 1:sz[1] - 1, 0:sz[2] - 2], yp[:, 1:sz[1] - 1, 1:sz[2] - 1]))
        M_pred_11 = tf.add(M_pred_11,tf.minimum(yp[:, 1:sz[1] - 1, 1:sz[2] - 1], yp[:, 1:sz[1] - 1, 1:sz[2] - 1]))
        M_pred_12 = tf.add(M_pred_12,tf.minimum(yp[:, 1:sz[1] - 1, 2:sz[2]], yp[:, 1:sz[1] - 1, 1:sz[2] - 1]))
        M_pred_20 = tf.add(M_pred_20,tf.minimum(yp[:, 2:sz[1], 0:sz[2] - 2], yp[:, 1:sz[1] - 1, 1:sz[2] - 1]))
        M_pred_21 = tf.add(M_pred_21,tf.minimum(yp[:, 2:sz[1], 1:sz[2] - 1], yp[:, 1:sz[1] - 1, 1:sz[2] - 1]))
        M_pred_22 = tf.add(M_pred_22,tf.minimum(yp[:, 2:sz[1], 2:sz[2]], yp[:, 1:sz[1] - 1, 1:sz[2] - 1]))

    # M_pred_00 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    # M_pred_01 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    # M_pred_02 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    # M_pred_10 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    # M_pred_11 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    # M_pred_12 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    # M_pred_20 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    # M_pred_21 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    # M_pred_22 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 2:sz[2]  ,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)

    M_true = M_true_00 + M_true_01 + M_true_02 + M_true_10 + M_true_11 + M_true_12 + M_true_20 + M_true_21 + M_true_22
    M_pred = M_pred_00 + M_pred_01 + M_pred_02 + M_pred_10 + M_pred_11 + M_pred_12 + M_pred_20 + M_pred_21 + M_pred_22
    M_join = tf.multiply(M_pred_00, M_true_00) + tf.multiply(M_pred_01, M_true_01) + tf.multiply(M_pred_02, M_true_02) + tf.multiply(M_pred_10, M_true_10) + tf.multiply(M_pred_11, M_true_11) + tf.multiply(M_pred_12, M_true_12) + tf.multiply(M_pred_20, M_true_20) + tf.multiply(M_pred_21, M_true_21) + tf.multiply(M_pred_22, M_true_22)


    en_true = - tf.reduce_mean(tf.log(M_true))
    en_pred = - tf.reduce_mean(tf.log(M_pred))
    en_join = - tf.reduce_mean(tf.log(M_join))

    VoI = en_join * 2 - en_pred - en_true
    # return tf.reduce_mean(tf.reduce_sum(y_true,axis=-1))
    return VoI


def VoI_Seg_2D(y_true, y_pred):
    mv = tf.reduce_max(y_true)
    I_weight = tf.cond(tf.greater(mv, 50), lambda: tf.add(0.5, 0.5), lambda: tf.add(0.0, 0.0))
    S_weight = 1.0 - I_weight
    y_true=y_true * S_weight
    sz = K.shape(y_pred)
    M_true_00 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 2, 0:sz[2]-2,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_01 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 2, 1:sz[2]-1,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_02 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 0:sz[1] - 2, 2:sz[2],:]  , y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_10 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1] - 1, 0:sz[2]-2,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_11 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1] - 1, 1:sz[2]-1,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_12 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 1:sz[1] - 1, 2:sz[2],:]  , y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_20 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1]    , 0:sz[2]-2,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_21 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1]    , 1:sz[2]-1,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)
    M_true_22 = tf.clip_by_value(tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1]    , 2:sz[2]  ,:], y_true[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1), 0.00001,1)

    M_pred_00 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_01 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_02 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_10 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_11 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_12 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_20 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_21 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_22 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 2:sz[2]  ,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)

    M_true = M_true_00 + M_true_01 + M_true_02 + M_true_10 + M_true_11 + M_true_12 + M_true_20 + M_true_21 + M_true_22
    M_pred = M_pred_00 + M_pred_01 + M_pred_02 + M_pred_10 + M_pred_11 + M_pred_12 + M_pred_20 + M_pred_21 + M_pred_22
    M_join = tf.multiply(M_pred_00, M_true_00) + tf.multiply(M_pred_01, M_true_01) + tf.multiply(M_pred_02, M_true_02) + tf.multiply(M_pred_10, M_true_10) + tf.multiply(M_pred_11, M_true_11) + tf.multiply(M_pred_12, M_true_12) + tf.multiply(M_pred_20, M_true_20) + tf.multiply(M_pred_21, M_true_21) + tf.multiply(M_pred_22, M_true_22)


    en_true = - tf.reduce_mean(tf.log(M_true))
    en_pred = - tf.reduce_mean(tf.log(M_pred))
    en_join = - tf.reduce_mean(tf.log(M_join))

    VoI = en_join * 2 - en_pred - en_true
    # return tf.reduce_mean(tf.reduce_sum(y_true,axis=-1))
    return VoI

# def VoI_Im_2D(im, y_pred, sigma=0.1):
#     sz = K.shape(y_pred)
#     M_true_00 = tf.abs(im[:, 0:sz[1] - 2, 0:sz[2]-2] - im[:, 1:sz[1]-1, 1:sz[2]-1])
#     M_true_01 = tf.abs(im[:, 0:sz[1] - 2, 1:sz[2]-1] - im[:, 1:sz[1]-1, 1:sz[2]-1])
#     M_true_02 = tf.abs(im[:, 0:sz[1] - 2, 2:sz[2]]   - im[:, 1:sz[1]-1, 1:sz[2]-1])
#     M_true_10 = tf.abs(im[:, 1:sz[1] - 1, 0:sz[2]-2] - im[:, 1:sz[1]-1, 1:sz[2]-1])
#     M_true_11 = tf.abs(im[:, 1:sz[1] - 1, 1:sz[2]-1] - im[:, 1:sz[1]-1, 1:sz[2]-1])
#     M_true_12 = tf.abs(im[:, 1:sz[1] - 1, 2:sz[2]]   - im[:, 1:sz[1]-1, 1:sz[2]-1])
#     M_true_20 = tf.abs(im[:, 2:sz[1]    , 0:sz[2]-2] - im[:, 1:sz[1]-1, 1:sz[2]-1])
#     M_true_21 = tf.abs(im[:, 2:sz[1]    , 1:sz[2]-1] - im[:, 1:sz[1]-1, 1:sz[2]-1])
#     M_true_22 = tf.abs(im[:, 2:sz[1]    , 2:sz[2]]   - im[:, 1:sz[1]-1, 1:sz[2]-1])
#     # M_true_00 = tf.reduce_sum(tf.abs(im[:, 0:sz[1] - 2, 0:sz[2]-2, :] - im[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     # M_true_01 = tf.reduce_sum(tf.abs(im[:, 0:sz[1] - 2, 1:sz[2]-1, :] - im[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     # M_true_02 = tf.reduce_sum(tf.abs(im[:, 0:sz[1] - 2, 2:sz[2], :]   - im[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     # M_true_10 = tf.reduce_sum(tf.abs(im[:, 1:sz[1] - 1, 0:sz[2]-2, :] - im[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     # M_true_11 = tf.reduce_sum(tf.abs(im[:, 1:sz[1] - 1, 1:sz[2]-1, :] - im[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     # M_true_12 = tf.reduce_sum(tf.abs(im[:, 1:sz[1] - 1, 2:sz[2], :]   - im[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     # M_true_20 = tf.reduce_sum(tf.abs(im[:, 2:sz[1]    , 0:sz[2]-2, :] - im[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     # M_true_21 = tf.reduce_sum(tf.abs(im[:, 2:sz[1]    , 1:sz[2]-1, :] - im[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     # M_true_22 = tf.reduce_sum(tf.abs(im[:, 2:sz[1]    , 2:sz[2]  , :] - im[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#
#     M_true_00 = tf.exp(-M_true_00 * sigma)
#     M_true_01 = tf.exp(-M_true_01 * sigma)
#     M_true_02 = tf.exp(-M_true_02 * sigma)
#     M_true_10 = tf.exp(-M_true_10 * sigma)
#     M_true_11 = tf.exp(-M_true_11 * sigma)
#     M_true_12 = tf.exp(-M_true_12 * sigma)
#     M_true_20 = tf.exp(-M_true_20 * sigma)
#     M_true_21 = tf.exp(-M_true_21 * sigma)
#     M_true_22 = tf.exp(-M_true_22 * sigma)
#
#     M_pred_00 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     M_pred_01 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     M_pred_02 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     M_pred_10 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     M_pred_11 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     M_pred_12 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     M_pred_20 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     M_pred_21 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#     M_pred_22 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 2:sz[2]  ,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
#
#     M_true = M_true_00 + M_true_01 + M_true_02 + M_true_10 + M_true_11 + M_true_12 + M_true_20 + M_true_21 + M_true_22
#     M_pred = M_pred_00 + M_pred_01 + M_pred_02 + M_pred_10 + M_pred_11 + M_pred_12 + M_pred_20 + M_pred_21 + M_pred_22
#     M_join = tf.multiply(M_pred_00, M_true_00) + tf.multiply(M_pred_01, M_true_01) + tf.multiply(M_pred_02, M_true_02) + tf.multiply(M_pred_10, M_true_10) + tf.multiply(M_pred_11, M_true_11) + tf.multiply(M_pred_12, M_true_12) + tf.multiply(M_pred_20, M_true_20) + tf.multiply(M_pred_21, M_true_21) + tf.multiply(M_pred_22, M_true_22)
#
#
#     en_true = - tf.reduce_mean(tf.log(M_true))
#     en_pred = - tf.reduce_mean(tf.log(M_pred))
#     en_join = - tf.reduce_mean(tf.log(M_join))
#
#     VoI = en_join * 2 - en_pred - en_true
#     return VoI

def VoI_Im_2D(y_true, y_pred, sigma=0.1):
    sz = K.shape(y_pred)
    imind=0
    M_true_00 = tf.abs(y_true[:, 0:sz[1] - 2, 0:sz[2]-2,imind] - y_true[:, 1:sz[1]-1, 1:sz[2]-1, imind])
    M_true_01 = tf.abs(y_true[:, 0:sz[1] - 2, 1:sz[2]-1,imind] - y_true[:, 1:sz[1]-1, 1:sz[2]-1, imind])
    M_true_02 = tf.abs(y_true[:, 0:sz[1] - 2, 2:sz[2],imind]   - y_true[:, 1:sz[1]-1, 1:sz[2]-1, imind])
    M_true_10 = tf.abs(y_true[:, 1:sz[1] - 1, 0:sz[2]-2,imind] - y_true[:, 1:sz[1]-1, 1:sz[2]-1, imind])
    M_true_11 = tf.abs(y_true[:, 1:sz[1] - 1, 1:sz[2]-1,imind] - y_true[:, 1:sz[1]-1, 1:sz[2]-1, imind])
    M_true_12 = tf.abs(y_true[:, 1:sz[1] - 1, 2:sz[2],imind]   - y_true[:, 1:sz[1]-1, 1:sz[2]-1, imind])
    M_true_20 = tf.abs(y_true[:, 2:sz[1]    , 0:sz[2]-2,imind] - y_true[:, 1:sz[1]-1, 1:sz[2]-1, imind])
    M_true_21 = tf.abs(y_true[:, 2:sz[1]    , 1:sz[2]-1,imind] - y_true[:, 1:sz[1]-1, 1:sz[2]-1, imind])
    M_true_22 = tf.abs(y_true[:, 2:sz[1]    , 2:sz[2]  ,imind] - y_true[:, 1:sz[1]-1, 1:sz[2]-1, imind])

    M_true_00 = tf.exp(-M_true_00 * sigma)
    M_true_01 = tf.exp(-M_true_01 * sigma)
    M_true_02 = tf.exp(-M_true_02 * sigma)
    M_true_10 = tf.exp(-M_true_10 * sigma)
    M_true_11 = tf.exp(-M_true_11 * sigma)
    M_true_12 = tf.exp(-M_true_12 * sigma)
    M_true_20 = tf.exp(-M_true_20 * sigma)
    M_true_21 = tf.exp(-M_true_21 * sigma)
    M_true_22 = tf.exp(-M_true_22 * sigma)

    M_pred_00 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_01 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_02 = tf.reduce_sum(tf.minimum(y_pred[:, 0:sz[1] - 2, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_10 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_11 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_12 = tf.reduce_sum(tf.minimum(y_pred[:, 1:sz[1] - 1, 2:sz[2],:]  , y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_20 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 0:sz[2]-2,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_21 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 1:sz[2]-1,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)
    M_pred_22 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1]    , 2:sz[2]  ,:], y_pred[:, 1:sz[1]-1, 1:sz[2]-1, :]), axis=-1)

    M_true = M_true_00 + M_true_01 + M_true_02 + M_true_10 + M_true_11 + M_true_12 + M_true_20 + M_true_21 + M_true_22
    M_pred = M_pred_00 + M_pred_01 + M_pred_02 + M_pred_10 + M_pred_11 + M_pred_12 + M_pred_20 + M_pred_21 + M_pred_22
    M_join = tf.multiply(M_pred_00, M_true_00) + tf.multiply(M_pred_01, M_true_01) + tf.multiply(M_pred_02, M_true_02) + tf.multiply(M_pred_10, M_true_10) + tf.multiply(M_pred_11, M_true_11) + tf.multiply(M_pred_12, M_true_12) + tf.multiply(M_pred_20, M_true_20) + tf.multiply(M_pred_21, M_true_21) + tf.multiply(M_pred_22, M_true_22)


    en_true = - tf.reduce_mean(tf.log(M_true))
    en_pred = - tf.reduce_mean(tf.log(M_pred))
    en_join = - tf.reduce_mean(tf.log(M_join))

    VoI = en_join * 2 - en_pred - en_true
    # return VoI / (tf.log(9.0) + en_join)
    return VoI

def VoI_2D(image_weight=1, sigma=0.1, nclass=2, ncluster=[]):
    def inner(y_true, y_pred):
        sz = K.shape(y_pred)
        mv=tf.reduce_max(y_true)
        I_weight = tf.cond(tf.greater(mv,50), lambda: tf.add(0.5,0.5), lambda: tf.add(0.0,0.0))
        S_weight = 1.0 - I_weight
        I_weight=I_weight*image_weight
        # I_weight = image_weight
        # seg_weight=1.0-image_weight
        # tf.cond(tf.greater(I_weight,0), lambda: print(LLL), lambda: tf.add(0.0,0.0))
        # return VoI_Seg_2D(y_true, y_pred) * S_weight
        # return VoI_Im_2D(y_true, y_pred, sigma=sigma) * I_weight + VoI_Seg_2D(y_true, y_pred) * S_weight

        return VoI_Im_2D(y_true, y_pred, sigma=sigma) * I_weight
    return inner


# def VoI_2D():
#     def inner(y_true, y_pred):
#         sz = K.shape(y_pred)
#         mv=tf.reduce_max(y_true)
#         I_weight = tf.cond(tf.greater(mv,50), lambda: tf.add(0.5,0.5), lambda: tf.add(0.0,0.0))
#         S_weight = 1.0 - I_weight
#         M_true_00 = tf.reduce_sum(
#             tf.minimum(y_true[:, 0:sz[1] - 2, 0:sz[2] - 2, :], y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_true_01 = tf.reduce_sum(
#             tf.minimum(y_true[:, 0:sz[1] - 2, 1:sz[2] - 1, :], y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_true_02 = tf.reduce_sum(
#             tf.minimum(y_true[:, 0:sz[1] - 2, 2:sz[2], :], y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_true_10 = tf.reduce_sum(
#             tf.minimum(y_true[:, 1:sz[1] - 1, 0:sz[2] - 2, :], y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_true_11 = tf.reduce_sum(
#             tf.minimum(y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :], y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_true_12 = tf.reduce_sum(
#             tf.minimum(y_true[:, 1:sz[1] - 1, 2:sz[2], :], y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_true_20 = tf.reduce_sum(
#             tf.minimum(y_true[:, 2:sz[1], 0:sz[2] - 2, :], y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_true_21 = tf.reduce_sum(
#             tf.minimum(y_true[:, 2:sz[1], 1:sz[2] - 1, :], y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_true_22 = tf.reduce_sum(tf.minimum(y_true[:, 2:sz[1], 2:sz[2], :], y_true[:, 1:sz[1] - 1, 1:sz[2] - 1, :]),
#                                   axis=-1)
#
#         M_pred_00 = tf.reduce_sum(
#             tf.minimum(y_pred[:, 0:sz[1] - 2, 0:sz[2] - 2, :], y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_pred_01 = tf.reduce_sum(
#             tf.minimum(y_pred[:, 0:sz[1] - 2, 1:sz[2] - 1, :], y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_pred_02 = tf.reduce_sum(
#             tf.minimum(y_pred[:, 0:sz[1] - 2, 2:sz[2], :], y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_pred_10 = tf.reduce_sum(
#             tf.minimum(y_pred[:, 1:sz[1] - 1, 0:sz[2] - 2, :], y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_pred_11 = tf.reduce_sum(
#             tf.minimum(y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :], y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_pred_12 = tf.reduce_sum(
#             tf.minimum(y_pred[:, 1:sz[1] - 1, 2:sz[2], :], y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_pred_20 = tf.reduce_sum(
#             tf.minimum(y_pred[:, 2:sz[1], 0:sz[2] - 2, :], y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_pred_21 = tf.reduce_sum(
#             tf.minimum(y_pred[:, 2:sz[1], 1:sz[2] - 1, :], y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :]), axis=-1)
#         M_pred_22 = tf.reduce_sum(tf.minimum(y_pred[:, 2:sz[1], 2:sz[2], :], y_pred[:, 1:sz[1] - 1, 1:sz[2] - 1, :]),
#                                   axis=-1)
#
#         M_true = M_true_00 + M_true_01 + M_true_02 + M_true_10 + M_true_11 + M_true_12 + M_true_20 + M_true_21 + M_true_22 + 0.001
#         M_pred = M_pred_00 + M_pred_01 + M_pred_02 + M_pred_10 + M_pred_11 + M_pred_12 + M_pred_20 + M_pred_21 + M_pred_22 + 0.001
#         M_join = tf.multiply(M_pred_00, M_true_00) + tf.multiply(M_pred_01, M_true_01) + tf.multiply(M_pred_02,
#                                                                                                      M_true_02) + tf.multiply(
#             M_pred_10, M_true_10) + tf.multiply(M_pred_11, M_true_11) + tf.multiply(M_pred_12, M_true_12) + tf.multiply(
#             M_pred_20, M_true_20) + tf.multiply(M_pred_21, M_true_21) + tf.multiply(M_pred_22, M_true_22)+0.001
#
#         en_true = - tf.reduce_mean(tf.log(M_true))
#         en_pred = - tf.reduce_mean(tf.log(M_pred))
#         en_join = - tf.reduce_mean(tf.log(M_join))
#
#         VoI_Seg = en_join * 2 - en_pred - en_true
#         return VoI
#     return inner



# def figure_weight_2D(exp=1.0):
#     def inner(y_true, y_pred):
#         sz = tf.shape(y_true)
#
#         y_pred = tf.reshape(y_pred, [sz[0] * sz[1], sz[2]])
#         y_true = tf.reshape(y_true, [sz[0] * sz[1], sz[2]])
#
#         # print(sess.run(y_true))
#
#         y_pred_bg = y_pred[:, 0]
#         y_true_bg = y_true[:, 0]
#
#         fg_inds = tf.where(tf.equal(y_true_bg, 0))
#         fg_N = tf.size(fg_inds)
#
#         bg_error = tf.abs(y_pred_bg - y_true_bg)
#
#         sorted_inds = tf.nn.top_k(bg_error, k=fg_N)
#         bg_inds = sorted_inds.indices
#         entropy_bg = tf.gather(y_true, bg_inds) * K.pow(-K.log(tf.gather(y_pred, bg_inds)), exp)
#         entropy_fg = tf.gather(y_true, fg_inds) * K.pow(-K.log(tf.gather(y_pred, fg_inds)), exp)
#         entropy = K.sum(entropy_bg, axis=-1) + K.sum(entropy_fg, axis=-1)
#         if K.ndim(entropy) == 2:
#             entropy = K.mean(entropy, axis=-1)
#         return entropy
#
#     YY=tf.concat([y_true,y_pred],1)
#     entropy=tf.map_fn(inner, YY, dtype="float")
#     # print(entropy)
#     cost=tf.reduce_mean(entropy)
#     return cost


def figure_weight_2D(exp=1.0,N=1.0):
    """
    :param exp: exponent. 1.0 for no exponential effect.
    """

    def inner(y_true, y_pred):
        sz = tf.shape(y_true)
        my_pred = tf.reshape(y_pred, [sz[0] * sz[1] * sz[2], sz[3]])
        my_true = tf.reshape(y_true, [sz[0] * sz[1] * sz[2], sz[3]])
        my_pred = K.clip(my_pred, K.epsilon(), 1 - K.epsilon())  # As log is used

        # print(sess.run(y_true))

        y_pred_bg = my_pred[:, 0]
        y_true_bg = my_true[:, 0]

        fg_inds = tf.where(tf.equal(y_true_bg, 0))
        # bg_inds = tf.where(tf.not_equal(y_true_bg, -1))
        fg_N = tf.size(fg_inds)
        Nsample=tf.multiply(fg_N,tf.to_int32(N))
        # print(sess.run(fg_inds))
        # print(fg_N)
        bg_error = tf.abs(y_pred_bg - y_true_bg)
        # print(sess.run(bg_error))

        sorted_inds = tf.nn.top_k(bg_error, k=Nsample)
        # inds = fg_inds
        # print(sess.run(sorted_inds))
        bg_inds = sorted_inds.indices
        entropy_bg = tf.gather(my_true[:,0], bg_inds) * K.pow(-K.log(tf.gather(my_pred[:,0], bg_inds)), exp)
        entropy_fg = tf.gather(my_true[:,0], fg_inds) * K.pow(-K.log(tf.gather(my_pred[:,0], fg_inds)), exp)
        entropy_bg = entropy_bg+tf.gather(my_true[:,1], bg_inds) * K.pow(-K.log(tf.gather(my_pred[:,1], bg_inds)), exp)
        entropy_fg = entropy_fg+tf.gather(my_true[:,1], fg_inds) * K.pow(-K.log(tf.gather(my_pred[:,1], fg_inds)), exp)
        entropy_0 = my_true * K.pow(-K.log(my_pred), exp)
        entropy_0 = K.mean(entropy_0)
        entropy=entropy_0*0.1 + (K.sum(entropy_bg)+K.sum(entropy_fg))/tf.to_float(Nsample+fg_N)
        # if K.ndim(entropy) == 2:
        #     entropy = K.mean(entropy, axis=-1)
        return entropy
    return inner

def get_channel_axis():
    """Gets the channel axis."""
    if K.image_data_format() == 'channels_first':
        return 1
    else:
        return -1

def VoI(NLabel=2, batch_size=1):
    """
    :param exp: exponent. 1.0 for no exponential effect.
    """

    def inner(y_true, y_pred):
        VoIs = 0
    # for i in range(batch_size):
    #     one_true = tf.squeeze(y_true[i, :])
    #     one_pred = tf.squeeze(y_pred[i, :])
        seg_true = tf.argmax(y_true, get_channel_axis())
        seg_pred = tf.argmax(y_pred, get_channel_axis())
        cc_true = tf.Variable(initial_value=tf.zeros(shape=[2, 1024,1024], dtype=tf.int32))
        cc_pred = tf.Variable(initial_value=tf.zeros(shape=[2, 1024,1024], dtype=tf.int32))

        for L in range(NLabel):
            mask = tf.equal(seg_pred, L)
            cc = tf.contrib.image.connected_components(mask)
            inds = tf.where(tf.not_equal(cc, 0))
            maxcc = tf.reduce_max(cc_pred)
            cc_pred = tf.scatter_nd_update(cc_pred, inds, tf.cast((tf.gather_nd(cc, inds)) + maxcc, tf.int32))

            mask = tf.equal(seg_true, L)
            cc = tf.contrib.image.connected_components(mask)
            inds = tf.where(tf.not_equal(cc, 0))
            maxcc = tf.reduce_max(cc_true)
            cc_true = tf.scatter_nd_update(cc_true, inds, tf.cast((tf.gather_nd(cc, inds)) + maxcc, tf.int32))

        maxcc_pred = tf.reduce_max(cc_pred)
        maxcc_true = tf.reduce_max(cc_true)
        hist_pred = tf.histogram_fixed_width(cc_pred, [0, maxcc_pred], maxcc_pred + 1)
        hist_joint = tf.histogram_fixed_width(cc_pred * (maxcc_true + 1) + cc_true,
                                              [0, maxcc_pred * (maxcc_true + 1)],
                                              maxcc_pred * (maxcc_true + 1) + 1)

        inds_pred = tf.where(tf.not_equal(hist_pred, 0))
        prob_pred = tf.divide(tf.gather_nd(hist_pred, inds_pred), tf.reduce_sum(hist_pred))

        inds_joint = tf.where(tf.not_equal(hist_joint, 0))
        prob_joint = tf.divide(tf.gather_nd(hist_joint, inds_joint), tf.reduce_sum(hist_joint))

        entropy_pred = -tf.reduce_sum(tf.multiply(prob_pred, tf.log(prob_pred)))
        entropy_joint = -tf.reduce_sum(tf.multiply(prob_joint, tf.log(prob_joint)))
        VoI = entropy_joint * 2 - entropy_pred
        VoIs = VoI + VoIs
        # print(sess.run(VoI))
        return VoIs
    return inner


# Using the equation of tf.nn.weighted_cross_entropy_with_logits
def exp_crossentropy(exp=1.0, ignore_neg=False):
    """
    :param exp: exponent. 1.0 for no exponential effect.
    :param ignore_neg: True if only the positives are considered.
    """

    def inner(y_true, y_pred):

        pos_weights = y_true * K.cast(K.greater(y_true, 0), K.floatx())
        neg_weights = - y_true * K.cast(K.less_equal(y_true, 0), K.floatx())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # As log is used

        if ignore_neg:
            entropy = pos_weights * K.pow(-K.log(y_pred), exp)
            entropy = K.sum(entropy, axis=-1)
        else:
            entropy = pos_weights * K.pow(-K.log(y_pred), exp) + neg_weights * K.pow(-K.log(1 - y_pred), exp)
            entropy = K.mean(entropy, axis=-1)

        if K.ndim(entropy) == 2:
            entropy = K.mean(entropy, axis=-1)

        return entropy

    return inner


def combine_loss(losses, weights):
    def inner(y_true, y_pred):
        output = 0.0
        for loss, w in zip(losses, weights):
            loss = loss(y_true, y_pred)
            if K.ndim(loss) == 2:
                loss = K.mean(loss, axis=-1)
            output += w * loss
        return output
    return inner
