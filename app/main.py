"""
app/main.py — unBiasFace FastAPI backend
"""

import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="unBiasFace")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load app data once at startup
with open(STATIC_DIR / "app_data.json") as f:
    APP_DATA = json.load(f)


@app.get("/", response_class=HTMLResponse)
def index():
    with open(BASE_DIR / "templates" / "index.html") as f:
        return f.read()


@app.get("/api/data")
def get_data():
    """Return full app data (structure only, no images) for the frontend."""
    # Return structure without images for initial load
    structure = {
        "tasks": {},
        "task_descriptions": APP_DATA["task_descriptions"],
        "axis_descriptions": APP_DATA["axis_descriptions"],
        "label_names": APP_DATA["label_names"],
    }
    for task, task_data in APP_DATA["tasks"].items():
        structure["tasks"][task] = {"axes": {}}
        for axis, axis_data in task_data["axes"].items():
            structure["tasks"][task]["axes"][axis] = {"subgroups": {}}
            for sg_val, sg_data in axis_data["subgroups"].items():
                structure["tasks"][task]["axes"][axis]["subgroups"][str(sg_val)] = {
                    "name": sg_data["name"],
                    "metrics": sg_data["metrics"],
                }
    return JSONResponse(structure)


@app.get("/api/images/{task}/{axis}/{subgroup_val}")
def get_images(task: str, axis: str, subgroup_val: str):
    """Return images for a specific task/axis/subgroup combination."""
    try:
        # Keys are stored as strings in the JSON
        images = APP_DATA["tasks"][task]["axes"][axis]["subgroups"][str(subgroup_val)]["images"]
        return JSONResponse({"images": images})
    except KeyError:
        return JSONResponse({"images": []}, status_code=404)
