#Created by Augmented Startups
#YOLOv7 Flask Application
#Enroll at www.augmentedstartups.com/store
from cProfile import label
from decimal import ROUND_HALF_UP, ROUND_UP
from wsgiref.validate import validator
from flask import Flask, render_template, Response,jsonify,request,session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
from flask_bootstrap import Bootstrap
import cv2

from hubconfCustom import video_detection
app = Flask(__name__)
Bootstrap(app)

app.config['SECRET_KEY'] = 'daniyalkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    # text = StringField(u'Conf: ', validators=[InputRequired()])
    conf_slide = IntegerRangeField('Confidence:  ', default=25,validators=[InputRequired()])
    submit = SubmitField("Run")
    



detect_count = 0
safe_count = 0

def generate_frames(path_x = '',conf_= 0.25):
    yolo_output = video_detection(path_x,conf_)
    global detect_count
    global safe_count
    for detection_, d_count,s_count in yolo_output:
        detect_count = str(d_count)
        safe_count = str(s_count)
        ref,buffer=cv2.imencode('.jpg',detection_)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')



@app.route("/",methods=['GET','POST'])
@app.route("/home", methods=['GET','POST'])

def home():
    session.clear()
    return render_template('root.html')


@app.route('/FrontPage',methods=['GET','POST'])
def front():
    # session.clear()
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        # conf_ = form.text.data
        conf_ = form.conf_slide.data
        
        # print(round(float(conf_)/100,2))
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        session['conf_'] = conf_
    return render_template('video.html',form=form)


@app.route('/video')
def video():
    return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detectionCount',methods = ['GET'])
def fps_fun():
    global detect_count
    return jsonify(detectCount=detect_count)

@app.route('/safeCount',methods = ['GET'])
def size_fun():
    global safe_count
    return jsonify(safecount=safe_count)







if __name__ == "__main__":
    app.run(debug=True)
