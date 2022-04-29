import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, Input, Dense, Lambda, concatenate, Conv2D, MaxPooling2D, Flatten,\
                                    BatchNormalization, Conv2D, Activation, GlobalAveragePooling2D, ZeroPadding2D, Add



##RESNET 50 LAYERS   https://eremo2002.tistory.com/76


class res1_layer(Layer):

    def __init__(self, **kwargs):
        super(res1_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv1_pad = ZeroPadding2D(padding=(3, 3))
        self.conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='valid')
        self.bn_conv1 = BatchNormalization()
        self.activation_1 = Activation('relu')
        self.pool1_pad = ZeroPadding2D(padding=(1, 1))

        super(res1_layer, self).build(input_shape)

    def call(self, inputs):

        x = self.conv1_pad(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.activation_1(x)
        x = self.pool1_pad(x)

        return x

class res2_layer(Layer):

    def __init__(self, **kwargs):
        super(res2_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.max_pooling2d_1 = MaxPooling2D((3, 3), 2)

        self.res2a_branch2a = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')
        self.bn2a_branch2a = BatchNormalization()
        self.activation_2 = Activation('relu')

        self.res2a_branch2b = Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2a_branch2b = BatchNormalization()
        self.activation_3 = Activation('relu')

        self.res2a_branch2c = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.res2a_branch1 = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn2a_branch2c = BatchNormalization()
        self.bn2a_branch1 = BatchNormalization()
        self.add_1 = Add()
        self.activation_4 = Activation('relu')


        self.res2b_branch2a = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')
        self.bn2b_branch2a = BatchNormalization()
        self.activation_5 = Activation('relu')

        self.res2b_branch2b = Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2b_branch2b = BatchNormalization()
        self.activation_6 = Activation('relu')

        self.res2b_branch2c = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn2b_branch2c = BatchNormalization()
        self.add_2 = Add()
        self.activation_7 = Activation('relu')


        self.res2c_branch2a = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')
        self.bn2c_branch2a = BatchNormalization()
        self.activation_8 = Activation('relu')

        self.res2c_branch2b = Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2c_branch2b = BatchNormalization()
        self.activation_9 = Activation('relu')

        self.res2c_branch2c = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn2c_branch2c = BatchNormalization()
        self.add_3 = Add()
        self.activation_10 = Activation('relu')

        super(res2_layer, self).build(input_shape)

    def call(self, inputs):

        x = self.max_pooling2d_1(inputs)

        shortcut = x

        for i in range(3):
            if (i == 0):
                x = self.res2a_branch2a(x)
                x = self.bn2a_branch2a(x)
                x = self.activation_2(x)

                x = self.res2a_branch2b(x)
                x = self.bn2a_branch2b(x)
                x = self.activation_3(x)

                x = self.res2a_branch2c(x)
                shortcut = self.res2a_branch1(shortcut)
                x = self.bn2a_branch2c(x)
                shortcut = self.bn2a_branch1(shortcut)

                x = self.add_1([x, shortcut])
                x = self.activation_4(x)

                shortcut = x

            elif (i == 1):
                x = self.res2b_branch2a(x)
                x = self.bn2b_branch2a(x)
                x = self.activation_5(x)

                x = self.res2b_branch2b(x)
                x = self.bn2b_branch2b(x)
                x = self.activation_6(x)

                x = self.res2b_branch2c(x)
                x = self.bn2b_branch2c(x)

                x = self.add_2([x, shortcut])
                x = self.activation_7(x)

                shortcut = x

            elif (i == 2):
                x = self.res2c_branch2a(x)
                x = self.bn2c_branch2a(x)
                x = self.activation_8(x)

                x = self.res2c_branch2b(x)
                x = self.bn2c_branch2b(x)
                x = self.activation_9(x)

                x = self.res2c_branch2c(x)
                x = self.bn2c_branch2c(x)

                x = self.add_3([x, shortcut])
                x = self.activation_10(x)

                shortcut = x

        return x

class res3_layer(Layer):

    def __init__(self, **kwargs):
        super(res3_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.res3a_branch2a = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')
        self.bn3a_branch2a = BatchNormalization()
        self.activation_11 = Activation('relu')

        self.res3a_branch2b = Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3a_branch2b = BatchNormalization()
        self.activation_12 = Activation('relu')

        self.res3a_branch2c = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.res3a_branch1 = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')
        self.bn3a_branch2c = BatchNormalization()
        self.bn3a_branch1 = BatchNormalization()
        self.add_4 = Add()
        self.activation_13 = Activation('relu')


        self.res3b_branch2a = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')
        self.bn3b_branch2a = BatchNormalization()
        self.activation_14 = Activation('relu')

        self.res3b_branch2b = Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3b_branch2b = BatchNormalization()
        self.activation_15 =Activation('relu')

        self.res3b_branch2c = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn3b_branch2c = BatchNormalization()
        self.add_5 = Add()
        self.activation_16 = Activation('relu')


        self.res3c_branch2a = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')
        self.bn3c_branch2a = BatchNormalization()
        self.activation_17 = Activation('relu')

        self.res3c_branch2b = Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3c_branch2b = BatchNormalization()
        self.activation_18 =Activation('relu')

        self.res3c_branch2c = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn3c_branch2c = BatchNormalization()
        self.add_6 = Add()
        self.activation_19 = Activation('relu')


        self.res3d_branch2a = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')
        self.bn3d_branch2a = BatchNormalization()
        self.activation_20 = Activation('relu')

        self.res3d_branch2b = Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3d_branch2b = BatchNormalization()
        self.activation_21 =Activation('relu')

        self.res3d_branch2c = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn3d_branch2c = BatchNormalization()
        self.add_7 = Add()
        self.activation_22 = Activation('relu')

        super(res3_layer, self).build(input_shape)

    def call(self, x):

        shortcut = x

        for i in range(4):
            if (i == 0):
                x = self.res3a_branch2a(x)
                x = self.bn3a_branch2a(x)
                x = self.activation_11(x)

                x = self.res3a_branch2b(x)
                x = self.bn3a_branch2b(x)
                x = self.activation_12(x)

                x = self.res3a_branch2c(x)
                shortcut = self.res3a_branch1(shortcut)
                x = self.bn3a_branch2c(x)
                shortcut = self.bn3a_branch1(shortcut)

                x = self.add_4([x, shortcut])
                x = self.activation_13(x)

                shortcut = x

            elif (i == 1):
                x = self.res3b_branch2a(x)
                x = self.bn3b_branch2a(x)
                x = self.activation_14(x)

                x = self.res3b_branch2b(x)
                x = self.bn3b_branch2b(x)
                x = self.activation_15(x)

                x = self.res3b_branch2c(x)
                x = self.bn3b_branch2c(x)

                x = self.add_5([x, shortcut])
                x = self.activation_16(x)

                shortcut = x

            elif (i == 2):
                x = self.res3c_branch2a(x)
                x = self.bn3c_branch2a(x)
                x = self.activation_17(x)

                x = self.res3c_branch2b(x)
                x = self.bn3c_branch2b(x)
                x = self.activation_18(x)

                x = self.res3c_branch2c(x)
                x = self.bn3c_branch2c(x)

                x = self.add_6([x, shortcut])
                x = self.activation_19(x)

                shortcut = x

            elif (i == 3):
                x = self.res3d_branch2a(x)
                x = self.bn3d_branch2a(x)
                x = self.activation_20(x)

                x = self.res3d_branch2b(x)
                x = self.bn3d_branch2b(x)
                x = self.activation_21(x)

                x = self.res3d_branch2c(x)
                x = self.bn3d_branch2c(x)

                x = self.add_7([x, shortcut])
                x = self.activation_22(x)

                shortcut = x

        return x

class res4_layer(Layer):

    def __init__(self, **kwargs):
        super(res4_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.res4a_branch2a = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')
        self.bn4a_branch2a = BatchNormalization()
        self.activation_23 = Activation('relu')

        self.res4a_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4a_branch2b = BatchNormalization()
        self.activation_24 = Activation('relu')

        self.res4a_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.res4a_branch1 = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')
        self.bn4a_branch2c = BatchNormalization()
        self.bn4a_branch1 = BatchNormalization()
        self.add_8 = Add()
        self.activation_25 = Activation('relu')


        self.res4b_branch2a = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn4b_branch2a = BatchNormalization()
        self.activation_26 = Activation('relu')

        self.res4b_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4b_branch2b = BatchNormalization()
        self.activation_27 =Activation('relu')

        self.res4b_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.bn4b_branch2c = BatchNormalization()
        self.add_9 = Add()
        self.activation_28 = Activation('relu')

        self.res4b_branch2a = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn4b_branch2a = BatchNormalization()
        self.activation_26 = Activation('relu')

        self.res4b_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4b_branch2b = BatchNormalization()
        self.activation_27 =Activation('relu')

        self.res4b_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.bn4b_branch2c = BatchNormalization()
        self.add_9 = Add()
        self.activation_28 = Activation('relu')


        self.res4c_branch2a = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn4c_branch2a = BatchNormalization()
        self.activation_29 = Activation('relu')

        self.res4c_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4c_branch2b = BatchNormalization()
        self.activation_30 =Activation('relu')

        self.res4c_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.bn4c_branch2c = BatchNormalization()
        self.add_10 = Add()
        self.activation_31 = Activation('relu')


        self.res4d_branch2a = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn4d_branch2a = BatchNormalization()
        self.activation_32 = Activation('relu')

        self.res4d_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4d_branch2b = BatchNormalization()
        self.activation_33 =Activation('relu')

        self.res4d_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.bn4d_branch2c = BatchNormalization()
        self.add_11 = Add()
        self.activation_34 = Activation('relu')


        self.res4e_branch2a = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn4e_branch2a = BatchNormalization()
        self.activation_35 = Activation('relu')

        self.res4e_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4e_branch2b = BatchNormalization()
        self.activation_36 =Activation('relu')

        self.res4e_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.bn4e_branch2c = BatchNormalization()
        self.add_12 = Add()
        self.activation_37 = Activation('relu')


        self.res4f_branch2a = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn4f_branch2a = BatchNormalization()
        self.activation_38 = Activation('relu')

        self.res4f_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4f_branch2b = BatchNormalization()
        self.activation_39 =Activation('relu')

        self.res4f_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.bn4f_branch2c = BatchNormalization()
        self.add_13 = Add()
        self.activation_40 = Activation('relu')

        super(res4_layer, self).build(input_shape)

    def call(self, x):

        shortcut = x

        for i in range(6):
            if (i == 0):
                x = self.res4a_branch2a(x)
                x = self.bn4a_branch2a(x)
                x = self.activation_23(x)

                x = self.res4a_branch2b(x)
                x = self.bn4a_branch2b(x)
                x = self.activation_24(x)

                x = self.res4a_branch2c(x)
                shortcut = self.res4a_branch1(shortcut)
                x = self.bn4a_branch2c(x)
                shortcut = self.bn4a_branch1(shortcut)

                x = self.add_8([x, shortcut])
                x = self.activation_25(x)

                shortcut = x

            elif (i == 1):
                x = self.res4b_branch2a(x)
                x = self.bn4b_branch2a(x)
                x = self.activation_26(x)

                x = self.res4b_branch2b(x)
                x = self.bn4b_branch2b(x)
                x = self.activation_27(x)

                x = self.res4b_branch2c(x)
                x = self.bn4b_branch2c(x)

                x = self.add_9([x, shortcut])
                x = self.activation_28(x)

                shortcut = x

            elif (i == 2):
                x = self.res4c_branch2a(x)
                x = self.bn4c_branch2a(x)
                x = self.activation_29(x)

                x = self.res4c_branch2b(x)
                x = self.bn4c_branch2b(x)
                x = self.activation_30(x)

                x = self.res4c_branch2c(x)
                x = self.bn4c_branch2c(x)

                x = self.add_10([x, shortcut])
                x = self.activation_31(x)

                shortcut = x

            elif (i == 3):
                x = self.res4d_branch2a(x)
                x = self.bn4d_branch2a(x)
                x = self.activation_32(x)

                x = self.res4d_branch2b(x)
                x = self.bn4d_branch2b(x)
                x = self.activation_33(x)

                x = self.res4d_branch2c(x)
                x = self.bn4d_branch2c(x)

                x = self.add_11([x, shortcut])
                x = self.activation_34(x)

                shortcut = x

            elif (i == 4):
                x = self.res4e_branch2a(x)
                x = self.bn4e_branch2a(x)
                x = self.activation_35(x)

                x = self.res4e_branch2b(x)
                x = self.bn4e_branch2b(x)
                x = self.activation_36(x)

                x = self.res4e_branch2c(x)
                x = self.bn4e_branch2c(x)

                x = self.add_12([x, shortcut])
                x = self.activation_37(x)

                shortcut = x

            elif (i == 5):
                x = self.res4f_branch2a(x)
                x = self.bn4f_branch2a(x)
                x = self.activation_38(x)

                x = self.res4f_branch2b(x)
                x = self.bn4f_branch2b(x)
                x = self.activation_39(x)

                x = self.res4f_branch2c(x)
                x = self.bn4f_branch2c(x)

                x = self.add_13([x, shortcut])
                x = self.activation_40(x)

                shortcut = x

        return x

class res5_layer(Layer):

    def __init__(self, **kwargs):
        super(res5_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.res5a_branch2a = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')
        self.bn5a_branch2a = BatchNormalization()
        self.activation_41 = Activation('relu')

        self.res5a_branch2b = Conv2D(521, (3, 3), strides=(1, 1), padding='same')
        self.bn5a_branch2b = BatchNormalization()
        self.activation_42 = Activation('relu')

        self.res5a_branch2c = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')
        self.res5a_branch1 = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')
        self.bn5a_branch2c = BatchNormalization()
        self.bn5a_branch1 = BatchNormalization()
        self.add_14 = Add()
        self.activation_43 = Activation('relu')


        self.res5b_branch2a = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn5b_branch2a = BatchNormalization()
        self.activation_44 = Activation('relu')

        self.res5b_branch2b = Conv2D(512, (3, 3), strides=(1, 1), padding='same')
        self.bn5b_branch2b = BatchNormalization()
        self.activation_45 =Activation('relu')

        self.res5b_branch2c = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')
        self.bn5b_branch2c = BatchNormalization()
        self.add_15 = Add()
        self.activation_46 = Activation('relu')


        self.res5c_branch2a = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn5c_branch2a = BatchNormalization()
        self.activation_47 = Activation('relu')

        self.res5c_branch2b = Conv2D(512, (3, 3), strides=(1, 1), padding='same')
        self.bn5c_branch2b = BatchNormalization()
        self.activation_48 =Activation('relu')

        self.res5c_branch2c = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')
        self.bn5c_branch2c = BatchNormalization()
        self.add_16 = Add()
        self.activation_49 = Activation('relu')

        super(res5_layer, self).build(input_shape)

    def call(self, x):

        shortcut = x

        for i in range(3):
            if (i == 0):
                x = self.res5a_branch2a(x)
                x = self.bn5a_branch2a(x)
                x = self.activation_41(x)

                x = self.res5a_branch2b(x)
                x = self.bn5a_branch2b(x)
                x = self.activation_42(x)

                x = self.res5a_branch2c(x)
                shortcut = self.res5a_branch1(shortcut)
                x = self.bn5a_branch2c(x)
                shortcut = self.bn5a_branch1(shortcut)

                x = self.add_14([x, shortcut])
                x = self.activation_43(x)

                shortcut = x

            elif (i == 1):
                x = self.res5b_branch2a(x)
                x = self.bn5b_branch2a(x)
                x = self.activation_44(x)

                x = self.res5b_branch2b(x)
                x = self.bn5b_branch2b(x)
                x = self.activation_45(x)

                x = self.res5b_branch2c(x)
                x = self.bn5b_branch2c(x)

                x = self.add_15([x, shortcut])
                x = self.activation_46(x)

                shortcut = x

            elif (i == 2):
                x = self.res5c_branch2a(x)
                x = self.bn5c_branch2a(x)
                x = self.activation_47(x)

                x = self.res5c_branch2b(x)
                x = self.bn5c_branch2b(x)
                x = self.activation_48(x)

                x = self.res5c_branch2c(x)
                x = self.bn5c_branch2c(x)

                x = self.add_16([x, shortcut])
                x = self.activation_49(x)

                shortcut = x

        return x
