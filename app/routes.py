from flask import redirect, request, render_template, jsonify
from . import application
import json
from app.ptc_vpf import proposition_classification


@application.route('/', methods=['GET', 'POST'])
def amf_ptc_vpf():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        ff = open(f.filename, 'r')
        content = json.load(ff)
        # Classify the I-nodes into Value-Fact-Policy classes
        response = proposition_classification(content)
        return jsonify(response)
    elif request.method == 'GET':
        return render_template('docs.html')
 
 
