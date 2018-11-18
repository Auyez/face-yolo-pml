import plaidml.keras
plaidml.keras.install_backend()
from model import create_model
from keras.models import load_model
from keras import optimizers, callbacks
from loss import yolo_loss
from generator import DataGenerator
import sys

model = create_model()
if len(sys.argv) > 1:
	print(sys.argv[1])
	model.load_weights(sys.argv[1])
adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
model.compile(loss=yolo_loss, optimizer = adam)
name = 'model.net'
if len(sys.argv) > 2:
	name = sys.argv[2]
model.save('models/' + name)
print('Saved model to {}'.format('models/' + name))
