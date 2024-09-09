from flask import Flask, request, jsonify
import json

def idk():
    mod = {
        "first":1,
        "second":2
    }
    return jsonify(mod)

test = idk()
print(json.loads(test.text))
