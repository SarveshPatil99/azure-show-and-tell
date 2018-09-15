# Flask imports
import os
from flask import Flask, render_template, send_from_directory, send_file, request, url_for, jsonify, redirect, Request, g
from io import StringIO
from werkzeug import secure_filename

# TTS imports
from tts import sapi
import pythoncom

# Show-and-Tell imports
import math
import heapq
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
from configuration import ModelConfig
from inference_wrapper import InferenceWrapper
from inference_utils.caption_generator import CaptionGenerator
from inference_utils.vocabulary import Vocabulary


FLAGS = tf.flags.FLAGS
slim = tf.contrib.slim

# Setting up the voice for TTS
voice = sapi.Sapi()


def del_all_flags(FLAGS):
    """
    Deletes all the flags to avoid the DuplicateFlagError
    """
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)
    return keys_list,flags_dict 
    

del_all_flags(FLAGS)
# Path to 
tf.flags.DEFINE_string("checkpoint_path", "./model.ckpt-2000000", "Model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "./word_counts.txt", "Text file containing the vocabulary.")
tf.logging.set_verbosity(tf.logging.INFO)
# Build the inference graph.
g = tf.Graph()
with g.as_default():
    model = InferenceWrapper()
    restore_fn = model.build_graph_from_config(ModelConfig(), FLAGS.checkpoint_path)
g.finalize()
vocab = Vocabulary(FLAGS.vocab_file)
sess = tf.InteractiveSession(graph=g)
restore_fn(sess)
generator = CaptionGenerator(model, vocab)
app=Flask(__name__)

os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
STATIC_FOLDER = os.path.join(app.root_path, 'static')
# Read count.txt and initialize count
with open('count.txt') as f:
    count  = int(next(f))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','PNG','JPEG','JPG'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = True
@app.route('/sandbox')
def sandbox():
    return send_from_directory('/static/sandbox', 'index.html')
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global count
    if request.method=='POST':
        file=request.files['file']
        if (file and allowed_file(file.filename)):
            filename=secure_filename(file.filename)
            uploadFilePath=os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(uploadFilePath)
            tf.flags.DEFINE_string("input_files", "./uploads/" + filename,"Image path.")
            with tf.gfile.GFile(FLAGS.input_files, "rb") as f:
                image = f.read()
                captions = generator.beam_search(sess, image)
                caps = []
                for i, caption in enumerate(captions):
                    # Ignore begin and end words.
                    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                    sentence = " ".join(sentence)
                    caps.append(sentence)
                FLAGS.__delattr__('input_files')
            pythoncom.CoInitialize()
            # Creates .wav file saying the caption
            voice.create_recording('output' + str(count) + '.wav', caps[0])
            count += 1
            # Update count.txt
            with open('count.txt', 'w') as f:
                f.write(str(count))
            return render_template('render.html', filename = 'uploads/' + filename, cap1 = caps[0], cap2 = caps[1], cap3 = caps[2])
    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)

@app.route('/show/<filename>')
def uploaded_file(filename):
    uploadFilePath=os.path.join(app.config['UPLOAD_FOLDER'],filename)
    return render_template('render.html',filename='/uploads/'+filename)
	
@app.route('/<filename>')
def download_file(filename):
    return send_from_directory(app.root_path,filename)

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port = 5000)