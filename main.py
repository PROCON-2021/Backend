from glob import glob
from time import sleep
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS
from multiprocessing import Array, Value, Process
from graph import graph_bin
from arduino import run
import numpy as np
from dnn_estimate import dnn_estimate

flag = Value('i', 0)
ch1 = Array('i', 3500)
ch2 = Array('i', 3500)
ch3 = Array('i', 3500)
proc = Process(target = run, args = [flag, ch1, ch2, ch3])

app = Flask(__name__, static_folder="../frontend/dist/static", template_folder="../frontend/dist")
CORS(app)

@app.route("/history/<key>", methods = ["GET"])
def history(key):
    files = glob("../csv/*.csv")
    if len(files) == 0:
        return jsonify({"history": "NOT FOUND"})
    
    path = files[-int(key) - 1]
    
    arr1, arr2, arr3 = [], [], []
    with open(path, mode='r') as f:
        lines = iter(f.readlines())
        score, mode, cl = map(int, next(lines).split(','))
        date = next(lines).strip()
        arr1 = list(map(int, next(lines).split(',')))
        arr2 = list(map(int, next(lines).split(',')))
        arr3 = list(map(int, next(lines).split(',')))

    graph = graph_bin(len(arr1), arr1, arr2, arr3)
    info = [ \
        ["膝を伸ばした状態を保てていません", "膝を上げれていません", "膝の上げ下げのスピードが速すぎます", "膝を伸ばす時に停止できていません", "膝が伸びきっていません", "正しい運動ができています"], \
        ["正しい運動ができています", "肩をついたままになっています", "上体を起こしすぎています", "反動がついています"], \
        ["正しい運動ができています", "腕を引きすぎています", "脇が開きすぎています", "体幹を保てていません"] \
      ][mode][cl]

    return jsonify({"count": len(files), "score": score, "info": info, "date": date, "ch1": graph[0], "ch2": graph[1], "ch3": graph[2]})

@app.route("/save/<m>", methods = ["GET"])
def save(m):
    if proc.is_alive():
        flag.value = 1
        while flag.value != 0:
            sleep(0.1)

        arr1, arr2, arr3 = [], [], []
        #腹筋のみ-300する
        if m == '1':
            arr1 = ch1[300:3499]
            arr2 = ch2[300:3499]
            arr3 = ch3[300:3499]
        else:
            arr1 = ch1[0:3499]
            arr2 = ch2[0:3499]
            arr3 = ch3[0:3499]
        
        sig = np.stack([arr1, arr2, arr3], axis=1)

        #DNN
        cl, score = dnn_estimate(sig, m)

        time = datetime.now()
        path = "../csv/" + time.strftime("%Y%m%d%H%M%S") + ".csv"
        with open(path, mode='w') as f:
            f.write(str(score))
            f.write(", ")
            f.write(m)
            f.write(", ")
            f.write(str(cl))
            f.write('\n')
            f.write(time.strftime("%Y/%m/%d %H:%M:%S"))
            f.write('\n')
            f.write(str(arr1)[1:-1])
            f.write('\n')
            f.write(str(arr2)[1:-1])
            f.write('\n')
            f.write(str(arr3)[1:-1])
            f.write('\n')
        return jsonify({"save": "Success!"})
    else:
        return jsonify({"save": "ERROR"})

if __name__ == "__main__":
    proc.start()
    app.run()
