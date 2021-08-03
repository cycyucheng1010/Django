from django.shortcuts import render,redirect
from .forms import uploadmodelform
from .models import photo
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
def index(request):
    #pic_obj = models.photo.objects.clear()
    # photos = photo.objects.all()  #查詢所有資料
    #preview=photo.objects.last() # 查詢最後一筆資料
    form=uploadmodelform()
    if request.method == "POST":
        form = uploadmodelform(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    context = {
        #'photos': photos,
        'form': form
        #'preview':preview
    }
    #return render(request, 'test3/unet_upload.html')


    return render(request,'test2/second.html',context)
def pred_demo(request):
	clf_name = [" LR "," MLP "," KNN "," RF "," SVM "]
	file_1='COVID119.jpg'
	file_path_1 = os.path.join('static/image_demo/',file_1)
	mask_file_path_1,masked_file_path_1,y_pred_name_mobile_1,y_pred_name_res_1,y_pred_name_dense_1,ensemble_results_1,m_1_1,m_1_2,m_1_3,r_1_1,r_1_2,r_1_3,d_1_1,d_1_2,d_1_3,prob_mobile_1,prob_res_1, prob_dense_1  = predict_f(file_1, file_path_1)
	
	file_2 = 'Pneumonia.jpg'
	file_path_2 = os.path.join('static/image_demo/', file_2)
	mask_file_path_2,masked_file_path_2, y_pred_name_mobile_2, y_pred_name_res_2,y_pred_name_dense_2,ensemble_results_2,m_2_1,m_2_2,m_2_3,r_2_1,r_2_2,r_2_3,d_2_1,d_2_2,d_2_3,prob_mobile_2,prob_res_2, prob_dense_2  = predict_f(file_2, file_path_2)

	file_3 = 'Normal.jpg'
	file_path_3 = os.path.join('static/image_demo/', file_3)
	mask_file_path_3,masked_file_path_3, y_pred_name_mobile_3, y_pred_name_res_3,y_pred_name_dense_3,ensemble_results_3,m_3_1,m_3_2,m_3_3,r_3_1,r_3_2,r_3_3,d_3_1,d_3_2,d_3_3,prob_mobile_3,prob_res_3, prob_dense_3 = predict_f(file_3, file_path_3)
	
	file_4 = 'dog_2.jpg'
	file_path_4 = os.path.join('static/image_demo/', file_4)
	mask_file_path_4,masked_file_path_4, y_pred_name_mobile_4, y_pred_name_res_4,y_pred_name_dense_4,ensemble_results_4,m_4_1,m_4_2,m_4_3,r_4_1,r_4_2,r_4_3,d_4_1,d_4_2,d_4_3,prob_mobile_4,prob_res_4, prob_dense_4 = predict_f(file_4, file_path_4)
	#1
	path1_0 = "image_demo/"+file_1
	path1_1 = mask_file_path_1
	path1_2 = masked_file_path_1
	path1_3=m_1_1
	path1_4=m_1_2
	path1_5=m_1_3
	path1_6=r_1_1
	path1_7=r_1_2
	path1_8=r_1_3
	path1_9=d_1_1
	path1_10=d_1_2
	path1_11=d_1_3
	pred_mobile = y_pred_name_mobile_1
	pred_res = y_pred_name_res_1
	pred_dense = y_pred_name_dense_1
	ensemble_results=ensemble_results_1
	prob_mobile=prob_mobile_1
	prob_res=prob_res_1
	prob_dense=prob_dense_1
	#2
	path2_0="image_demo/"+file_2
	path2_1=mask_file_path_2
	path2_2=masked_file_path_2
	path2_3=m_2_1
	path2_4=m_2_2
	path2_5=m_2_3
	path2_6=r_2_1
	path2_7=r_2_2
	path2_8=r_2_3
	path2_9=d_2_1
	path2_10=d_2_2
	path2_11=d_2_3
	pred_mobile_2=y_pred_name_mobile_2
	pred_res_2=y_pred_name_res_2
	pred_dense_2=y_pred_name_dense_2
	ensemble_results_2=ensemble_results_2
	prob_mobile_2=prob_mobile_2
	prob_res_2=prob_res_2
	prob_dense_2=prob_dense_2
	#3
	path3_0 = "image_demo/"+file_3
	path3_1 = mask_file_path_3
	path3_2 = masked_file_path_3
	path3_3=m_3_1
	path3_4=m_3_2
	path3_5=m_3_3
	path3_6=r_3_1
	path3_7=r_3_2
	path3_8=r_3_3
	path3_9=d_3_1
	path3_10=d_3_2
	path3_11=d_3_3
	pred_mobile_3 = y_pred_name_mobile_3
	pred_res_3 = y_pred_name_res_3
	pred_dense_3 = y_pred_name_dense_3
	ensemble_results_3 = ensemble_results_3
	prob_mobile_3=prob_mobile_3
	prob_res_3=prob_res_3
	prob_dense_3=prob_dense_3
	#4
	path4_0 = "image_demo/"+file_4
	path4_1 = mask_file_path_4
	path4_2= masked_file_path_4
	path4_3=m_4_1
	path4_4=m_4_2
	path4_5=m_4_3
	path4_6=r_4_1
	path4_7=r_4_2
	path4_8=r_4_3
	path4_9=d_4_1
	path4_10=d_4_2
	path4_11=d_4_3
	pred_mobile_4 = y_pred_name_mobile_4
	pred_res_4 = y_pred_name_res_4
	pred_dense_4 = y_pred_name_dense_4
	ensemble_results_4 = ensemble_results_4
	prob_mobile_4=prob_mobile_4
	prob_res_4=prob_res_4
	prob_dense_4=prob_dense_4

	context={
		'path1_0' : path1_0,
		'path1_1' : path1_1,
		'path1_2' : path1_2,
		'path1_3' : path1_3,
		'path1_4' : path1_4,
		'path1_5' : path1_5,
		'path1_6' : path1_6,
		'path1_7' : path1_7,
		'path1_8' : path1_8,
		'path1_9' : path1_9,
		'path1_10' : path1_10,
		'path1_11' : path1_11,
		'pred_mobile' : pred_mobile,
		'pred_res' : pred_res,
		'pred_dense' : pred_dense,
		'ensemble_results' : ensemble_results,
		'prob_mobile' : prob_mobile,
		'prob_res' : prob_res,
		'prob_dense' :prob_dense,

		'path2_0' : path2_0,
		'path2_1' : path2_1,
		'path2_2' : path2_2,
		'path2_3' : path2_3,
		'path2_4' : path2_4,
		'path2_5' : path2_5,
		'path2_6' : path2_6,
		'path2_7' : path2_7,
		'path2_8' : path2_8,
		'path2_9' : path2_9,
		'path2_10' : path2_10,
		'path2_11' : path2_11,
		'pred_mobile_2' : pred_mobile_2,
		'pred_res_2' : pred_res_2,
		'pred_dense_2' : pred_dense_2,
		'ensemble_results_2' : ensemble_results_2,
		'prob_mobile_2' : prob_mobile_2,
		'prob_res_2' : prob_res_2,
		'prob_dense_2' :prob_dense_2,

		'path3_0' : path3_0,
		'path3_1' : path3_1,
		'path3_2' : path3_2,
		'path3_3' : path3_3,
		'path3_4' : path3_4,
		'path3_5' : path3_5,
		'path3_6' : path3_6,
		'path3_7' : path3_7,
		'path3_8' : path3_8,
		'path3_9' : path3_9,
		'path3_10' : path3_10,
		'path3_11' : path3_11,
		'pred_mobile_3' : pred_mobile_3,
		'pred_res_3' : pred_res_3,
		'pred_dense_3' : pred_dense_3,
		'ensemble_results_3' : ensemble_results_3,
		'prob_mobile_3' : prob_mobile_3,
		'prob_res_3' : prob_res_3,
		'prob_dense_3' :prob_dense_3,

		'path4_0' : path4_0,
		'path4_1' : path4_1,
		'path4_2' : path4_2,
		'path4_3' : path4_3,
		'path4_4' : path4_4,
		'path4_5' : path4_5,
		'path4_6' : path4_6,
		'path4_7' : path4_7,
		'path4_8' : path4_8,
		'path4_9' : path4_9,
		'path4_10' : path4_10,
		'path4_11' : path4_11,
		'pred_mobile_4' : pred_mobile_4,
		'pred_res_4' : pred_res_4,
		'pred_dense_4' : pred_dense_4,
		'ensemble_results_4' : ensemble_results_4,
		'clf_name' : clf_name,
		'prob_mobile_4' : prob_mobile_4,
		'prob_res_4' : prob_res_4,
		'prob_dense_4' :prob_dense_4,

	}
	return render(request,'test3/unet_pred_demo.html',context)



import cv2


@csrf_exempt
def unet_pred(request):
	if request.method == "POST":
		form = uploadmodelform(request.POST, request.FILES)
		if form.is_valid():
			form.save()
	file=photo.objects.last() # 查詢最後一筆資料
	file_path=os.path.join('media/',file.image.name)#'/home/yucheng/Django/cjlee_medical_image/media/image/lung_3vKSDLq.jpeg'
	img = cv2.imread(file_path)
	new_name='input_image.jpg'
	new_file_path=os.path.join('static/image_org',new_name)
	cv2.imwrite(new_file_path,img)
	mask_file_path, masked_file_path, y_pred_name_mobile, y_pred_name_res,y_pred_name_dense, ensemble_results,m_1,m_2,m_3,r_1,r_2,r_3,d_1,d_2,d_3,prob_mobile, prob_res, prob_dense = predict_f(new_name, new_file_path)
	
	clf_name = [" LR "," MLP "," KNN "," RF "," SVM "]
	path0=new_name
	path1=mask_file_path
	path2=masked_file_path
	path3=m_1
	path4=m_2
	path5=m_3
	path6=r_1
	path7=r_2
	path8=r_3
	path9=d_1
	path10=d_2
	path11=d_3
	pred_mobile = y_pred_name_mobile
	pred_res = y_pred_name_res
	pred_dense = y_pred_name_dense
	context={
		'path0':path0,
		'path1':path1,
		'path2':path2,
		'path3':path3,
		'path4':path4,
		'path5':path5,
		'path6':path6,
		'path7':path7,
		'path8':path8,
		'path9':path9,
		'path10':path10,
		'path11':path11,
		'clf_name':clf_name,
		'ensemble_results':ensemble_results,
		'prob_mobile':prob_mobile,
		'prob_res':prob_res,
		'prob_dense':prob_dense,
		'pred_mobile':pred_mobile,
		'pred_res':pred_res,
		'pred_dense':pred_dense
	}
	return render(request,'test3/unet_pred.html',context)#'test3/unet_pred.html'

# justin's AI code
#from werkzeug.utils import secure_filename
from keras.applications.mobilenet import preprocess_input as preprocess_1
from keras.applications.resnet import preprocess_input as preprocess_2
from keras.applications.densenet import preprocess_input as preprocess_3
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from keras_unet_collection import utils
import cv2
from PIL import Image
import shutil

shutil.rmtree('static/image_org')
os.makedirs('static/image_org')

model_unet = tf.keras.models.load_model('/home/yucheng/Django/cjlee_medical_image/uploadfile/unet3plus_tri_210521_2ndaddition.h5')#,custom_objects={'Functional':keras.models.Model})
model_mobile = tf.keras.models.load_model('/home/yucheng/Django/cjlee_medical_image/uploadfile/mobilenet_newimg_pretrain_2dense_best.h5')
model_res = tf.keras.models.load_model('/home/yucheng/Django/cjlee_medical_image/uploadfile/ResNet152V2_newimg_2dense_best.h5')
model_dense = tf.keras.models.load_model('/home/yucheng/Django/cjlee_medical_image/uploadfile/DenseNet201_newimg_2dense_best.h5')

clf_list = [
    'chestLR_04July2021',
    'chestMLP_04July2021',
    'chestKNN_04July2021',
    'chestRF_04July2021',
    'chestSVM_04July2021'
]
#clf_name = [" LR "," MLP "," KNN "," RF "," SVM "]
class_names = ['Normal', 'Pneumonia', 'COVID-19']
#app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = 'static/image_org'
#app.config['demo_FOLDER'] = 'static/image_demo'

class GradCAM:
	# Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
	def __init__(self, model, layerName=None):
		"""
        model: pre-softmax layer (logit layer)
        """
		self.model = model
		self.layerName = layerName

		if self.layerName == None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		for layer in reversed(self.model.layers):
			if len(layer.output_shape) == 4:
				return layer.name
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

	def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
		gradModel = tf.keras.models.Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output, self.model.output]
		)
		# record operations for automatic differentiation

		with tf.GradientTape() as tape:
			inputs = tf.cast(image, tf.float32)
			(convOuts, preds) = gradModel(inputs)  # preds after softmax
			loss = preds[:, classIdx]

		# compute gradients with automatic differentiation
		grads = tape.gradient(loss, convOuts)
		# discard batch
		convOuts = convOuts[0]
		grads = grads[0]
		norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

		# compute weights
		weights = tf.reduce_mean(norm_grads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

		# Apply reLU
		cam = np.maximum(cam, 0)
		cam = cam / np.max(cam)
		cam = cv2.resize(cam, upsample_size, interpolation=cv2.INTER_LINEAR)

		# convert to 3D
		cam3 = np.expand_dims(cam, axis=2)
		cam3 = np.tile(cam3, [1, 1, 3])
		return cam3


def overlay_gradCAM(img, cam3):
	cam3 = np.uint8(255 * cam3)
	#     cam3 = 255*cam3
	#     print("cam3 1",cam3.shape,cam3[112,0:10])
	cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
	#     print("cam3 2",cam3.shape,cam3[112,0:10])
	new_img = 0.3 * cam3 + 0.5 * img

	return (new_img * 255.0 / new_img.max()).astype("uint8")

@tf.custom_gradient
def guidedRelu(x):
	def grad(dy):
		return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

	return tf.nn.relu(x), grad

class GuidedBackprop:
	def __init__(self, model, layerName=None):
		self.model = model
		self.layerName = layerName
		self.gbModel = self.build_guided_model()

		if self.layerName == None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		for layer in reversed(self.model.layers):
			if len(layer.output_shape) == 4:
				return layer.name
		raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

	def build_guided_model(self):
		gbModel = tf.keras.models.Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output]
		)
		layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer, "activation")]
		for layer in layer_dict:
			if layer.activation == tf.keras.activations.relu:
				layer.activation = guidedRelu

		return gbModel

	def guided_backprop(self, images, upsample_size):
		"""Guided Backpropagation method for visualizing input saliency."""
		with tf.GradientTape() as tape:
			inputs = tf.cast(images, tf.float32)
			tape.watch(inputs)
			outputs = self.gbModel(inputs)

		grads = tape.gradient(outputs, inputs)[0]

		saliency = cv2.resize(np.asarray(grads), upsample_size)

		return saliency

def deprocess_image(x):
	"""Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
	# normalize tensor: center on 0., ensure std is 0.25
	x = x.copy()
	x -= x.mean()
	x /= (x.std() + tf.keras.backend.epsilon())
	x *= 0.25

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	if tf.keras.backend.image_data_format() == 'channels_first':
		x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def show_gradCAMs(model, gradCAM, GuidedBP, img,img_orig):
	upsample_size = (img.shape[1], img.shape[2])
	preds = model.predict(img)
	idx = preds.argmax()

	cam3 = gradCAM.compute_heatmap(image=img, classIdx=idx, upsample_size=upsample_size)
	heatmap = overlay_gradCAM(img_orig, cam3) #圖1 heatmap
	# heatmap = overlay_gradCAM(img_orig, cam3)[..., ::-1] #圖1 heatmap

	#     Show guided GradCAM
	gb = GuidedBP.guided_backprop(img, upsample_size)
	gb_viz = gb - np.min(gb)
	gb_viz /= gb_viz.max() #圖2 gb
	gb_viz = (gb_viz*255).astype('uint8')

	guided_gradcam = deprocess_image(gb * cam3)
	guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB) #圖3 guided_gradcam
	return heatmap, gb_viz, guided_gradcam

##for unet pred.
def input_data_process(input_array):
    return input_array/255.

def predict_f(file, file_path):
	img = input_data_process(utils.image_to_array(np.array(list([file_path])), size=128, channel=3))
	print("hi new",img.shape,img[0,64,:10,0])
	img = cv2.cvtColor(img[0].astype('float32'), cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	img = np.expand_dims(img, axis=0)
	temp_out = model_unet.predict(img)
	y_pred = temp_out[-1]
	for i in range(64):
		a = y_pred[0, i, :, 2].copy()
		y_pred[0, i, :, 2] = y_pred[0, 127 - i, :, 2]
		y_pred[0, 127 - i, :, 2] = a

	mask = cv2.resize(y_pred[0, ..., 2].copy(), (480, 480))
	mask[mask > 0.1] = 255
	mask[mask <= 0.1] = 0
	mask_file_path = "image/"+"mask_"+file
	cv2.imwrite("static/"+mask_file_path, mask.astype('uint8'))

	img = cv2.imread(file_path)[...,::-1]
	print("Orig image size", img.shape)
	img = cv2.resize(img, (480, 480))
	print("1",img.shape)
	mask[mask >= 130] = 255
	mask[mask < 130] = 0
	img[mask == 0] = 0
	masked_file_path = "image/" + "masked_" + file
	cv2.imwrite("static/"+masked_file_path, img[...,::-1].astype('uint8'))


	img = cv2.resize(img, (224, 224))
	img_orig = img.copy()
	img = img.reshape(-1, 224, 224, 3)

	img_1 = preprocess_1(img.copy())  ####important attention
	Y_pred_mobile = model_mobile.predict(img_1)
	prob_mobile = Y_pred_mobile.copy().astype('float16')
	prob_mobile = 'Nor.prob : ' + str(np.around(prob_mobile[0,0],3)) + '   Pneu.prob :' + str(np.around(prob_mobile[0,1],3))+ '   COVID19.prob :' + str(np.around(prob_mobile[0,2],3))
	print(Y_pred_mobile.shape,Y_pred_mobile)
	y_pred_mobile = np.argmax(Y_pred_mobile, axis=1)
	y_pred_name_mobile = [class_names[i] for i in y_pred_mobile]

	img_2 = preprocess_2(img.copy())  ####important attention
	Y_pred_res = model_res.predict(img_2)
	prob_res = Y_pred_res.copy().astype('float16')
	prob_res = 'Nor.prob : ' + str(np.around(prob_res[0,0],3)) + '   Pneu.prob :' + str(np.around(prob_res[0,1],3))+ '   COVID19.prob :' + str(np.around(prob_res[0,2],3))
	y_pred_res = np.argmax(Y_pred_res, axis=1)
	y_pred_name_res = [class_names[i] for i in y_pred_res]

	img_3 = preprocess_3(img.copy())  ####important attention
	Y_pred_dense = model_dense.predict(img_3)
	prob_dense = Y_pred_dense.copy().astype('float16')
	prob_dense = 'Nor.prob : ' + str(np.around(prob_dense[0,0],3)) + '   Pneu.prob :' + str(np.around(prob_dense[0,1],3))+ '   COVID19.prob :' + str(np.around(prob_dense[0,2],3))
	y_pred_mobile_dense = np.argmax(Y_pred_dense, axis=1)
	y_pred_name_dense = [class_names[i] for i in y_pred_mobile_dense]

	ensemble_prob_list = np.concatenate((Y_pred_mobile, Y_pred_res, Y_pred_dense), axis=None).reshape(1,-1)
	print(ensemble_prob_list)
	ensemble_results = ensemble(ensemble_prob_list)
	print(ensemble_results)

	#Grad-CAM of MobileNet
	gradCAM = GradCAM(model=model_mobile, layerName="conv_pw_13_relu")
	guidedBP = GuidedBackprop(model=model_mobile, layerName="conv_pw_13_relu")
	heatmap, gb, guided_gradcam = show_gradCAMs(model_mobile, gradCAM, guidedBP, img_1,img_orig)
	cv2.imwrite("static/" +"image/" + "heatmap_M_" + file, heatmap)
	cv2.imwrite("static/" +"image/" + "gb_M_" + file, gb)
	cv2.imwrite("static/" +"image/" + "guided_gradcam_M_" + file, guided_gradcam)

	#Grad-CAM of ResNet
	gradCAM = GradCAM(model=model_res, layerName="conv5_block3_out")
	guidedBP = GuidedBackprop(model=model_res, layerName="conv5_block3_out")
	heatmap, gb, guided_gradcam = show_gradCAMs(model_res, gradCAM, guidedBP, img_2,img_orig)
	cv2.imwrite("static/" +"image/" + "heatmap_R_" + file, heatmap)
	cv2.imwrite("static/" +"image/" + "gb_R_" + file, gb)
	cv2.imwrite("static/" +"image/" + "guided_gradcam_R_" + file, guided_gradcam)

	#Grad-CAM of Dense
	gradCAM = GradCAM(model=model_dense, layerName="conv5_block32_concat")
	guidedBP = GuidedBackprop(model=model_dense, layerName="conv5_block32_concat")
	heatmap, gb, guided_gradcam = show_gradCAMs(model_dense, gradCAM, guidedBP, img_3,img_orig)
	cv2.imwrite("static/" +"image/" + "heatmap_D_" + file, heatmap)
	cv2.imwrite("static/" +"image/" + "gb_D_" + file, gb)
	cv2.imwrite("static/" +"image/" + "guided_gradcam_D_" + file, guided_gradcam)

	return (mask_file_path,masked_file_path, y_pred_name_mobile, y_pred_name_res,y_pred_name_dense,ensemble_results,
			"image/" + "heatmap_M_" + file,"image/" + "gb_M_" + file,"image/" + "guided_gradcam_M_" + file,
			"image/" + "heatmap_R_" + file,"image/" + "gb_R_" + file,"image/" + "guided_gradcam_R_" + file,
			"image/" + "heatmap_D_" + file,"image/" + "gb_D_" + file,"image/" + "guided_gradcam_D_" + file,
			prob_mobile,prob_res,prob_dense)


def ensemble(ensemble_prob_list):
	ensemble_pred_list = list()
	for clf_member in clf_list:
		# print("hi3",clf_member)
		temp=os.path.join('/home/yucheng/Django/cjlee_medical_image/uploadfile/',clf_member)
		clf = joblib.load(temp)
		# print("hi4")
		prediction = clf.predict(ensemble_prob_list).tolist()
		prediction = [class_names[i] for i in prediction]
		# print("Prediction",prediction)
		ensemble_pred_list.append(prediction)
	print(len(ensemble_pred_list),ensemble_pred_list)
	return ensemble_pred_list

