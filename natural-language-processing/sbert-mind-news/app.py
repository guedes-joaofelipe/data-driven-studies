import os
import requests
from flask import Flask, request, abort
from flask_cors import CORS
import chromadb

app = Flask(__name__)
CORS(app)

client = chromadb.PersistentClient(path="./db/")


def get_filters(args) -> dict:
    """Auxiliary function to extract filters from query string"""
    return {
        "$and": [
            {arg.replace("filter.", ""): args.get(arg)}
            for arg in request.args
            if "filter." in arg
        ]
    }


@app.get("/<string:collection>/similars")
def get_similars(collection: str):
    """Get similar documents based on seed id"""
    collection = client.get_collection(name=collection)
    ids = request.args.get("ids", default="", type=str)
    where = get_filters(request.args)

    if ids in ["", None]:
        abort(404)

    seed = collection.get(ids=ids)

    if len(seed["ids"]) == 0:
        return {}

    similars = collection.query(
        query_texts=seed["documents"],
        n_results=request.args.get("lim", default=10, type=int),
        include=["distances", "metadatas"],
        where=where,
    )

    return similars


@app.get("/<string:collection>/query")
def query(collection: str):
    collection = client.get_collection(name=collection)
    text = request.args.get("text", default="", type=str)
    lim = request.args.get("lim", default=10, type=int)
    where = get_filters(request.args)

    return collection.query(
        query_texts=[text],
        n_results=lim,
        include=["distances", "metadatas", "documents"],
        where=where,
    )


@app.get("/")
def ping():
    return {"heartbeat": client.heartbeat()}
