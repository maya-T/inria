from flask import Flask
from flask_restful import Api
from flask_restful import Resource


class Todo(Resource):
    def get(self, id):
        return "hello  world", 200

    def put(self, id):
        return "hello  world", 200


app = Flask(__name__)
api = Api(app)
api.add_resource(Todo, "/todo/<int:id>")

if __name__ == "__main__":
    app.debug = True
    app.run()
