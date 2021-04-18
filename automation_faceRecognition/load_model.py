# MIT License
# Copyright (c) 2021 Loyio

from keras_vggface import VGGFace

model = VGGFace(model='resnet50')

print('Inputs: %s'%model.inputs)
print('Outputs: %s'%model.outputs)