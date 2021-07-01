from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from predict import predit
import pickle as pkl
from log import get_lgger
app=Flask(__name__)




@app.route('/predict',methods=['GET','POST'])
def index():
    path = os.path.dirname(__file__)
    pathdir=os.path.join(path,'uploads/')
    try:
        for i in os.listdir(pathdir):
            os.remove(pathdir+str(i))
    except:
        t = get_lgger('INFO')
        t.info('Uploaded Images not deleted.')
    if request.method=='POST':
        try:
            t = get_lgger('INFO')
            t.info('logging started')
            content=request.files['file']
            path_dir=os.path.join(path,'uploads',secure_filename(content.filename))
            content.save(path_dir)
            print("image is ",str(path_dir))
            print("path")
            #Make predictions
            result_value=predit(str(path_dir))
            label_encoder=pkl.load(open('labencdr.pkl','rb'))
            #labels=['daisy','dandelion','rose','sunflower','tulip']
            result=label_encoder.inverse_transform(result_value)
            t.info('log done.')
            return render_template('predict.html',result=result) #"Namasthe"
        except Exception as e:
            t = get_lgger('ERROR')
            t.error(str(e))
            return render_template('predict.html',result=str(e))
    return render_template('predict.html')


if __name__=='__main__':
    app.run(host='0.0.0.0')