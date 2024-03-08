from flask import Flask, render_template

app = Flask(__name__)
@app.route('/',methods=['POST', 'GET'])
def home():
    return render_template('home.html')

@app.route('/workflow',methods=['POST', 'GET'])
def workflow():
    return render_template('workflow.html')

@app.route('/display',methods=['POST', 'GET'])
def display():
    return render_template('display.html')

if __name__ == '__main__':
    app.run(debug=True)