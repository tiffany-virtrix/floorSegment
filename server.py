from fastapi import FastAPI, status, File, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import zipfile
import numpy as np
from io import BytesIO
from PIL import Image
from base64 import b64encode, b64decode

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

sam_checkpoint = "sam_vit_l_0b3195.pth" # "sam_vit_l_0b3195.pth" or "sam_vit_h_4b8939.pth"
model_type = "vit_l" # "vit_l" or "vit_h"
device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
print("Finishing loading")
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

input_point = []
input_label = []
masks = []
mask_input = [None]

GLOBAL_IMAGE = None
GLOBAL_MASK = None
GLOBAL_ZIPBUFFER = None

@app.post("/image")
async def process_images(
    image: UploadFile = File(...)
):
    global input_point, input_label, mask_input, masks
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER

    input_point = []
    input_label = []
    masks = []
    # mask_input = [None]

    # Read the image and mask data as bytes
    image_data = await image.read()

    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data))
    print("get image", img.shape)
    GLOBAL_IMAGE = img[:,:,:-1]
    GLOBAL_MASK = None
    GLOBAL_ZIPBUFFER = None
    # produce an image embedding by calling SamPredictor.set_image
    predictor.set_image(GLOBAL_IMAGE)
    print("finish setting image")
    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
        },
        status_code=200,
    )


@app.post("/undo")
async def undo_mask():
    global input_point, input_label, mask_input
    input_point.pop()
    input_label.pop()
    masks.pop()
    # mask_input.pop()

    return JSONResponse(
        content={
            "message": "Clear successfully",
        },
        status_code=200,
    )

@app.post("/click")
async def click_images(
    x: int = Form(...), # horizontal
    y: int = Form(...)  # vertical
):  
    global input_point, input_label, mask_input
    input_point.append([x, y])
    input_label.append(1)
    print("get click", x, y)
    print("input_point", input_point)
    print("input_label", input_label)

    
    masks_, scores_, logits_ = predictor.predict(
        point_coords=np.array([input_point[-1]]),
        point_labels=np.array([input_label[-1]]),
        # mask_input=mask_input[-1],
        multimask_output=True, # SAM outputs 3 masks, we choose the one with highest score
    )
    
    # mask_input.append(logits[np.argmax(scores), :, :][None, :, :])
    masks.append(masks_[np.argmax(scores_), :, :])
    res = np.zeros(masks[0].shape)
    for mask in masks:
        res = np.logical_or(res, mask)
    res = Image.fromarray(res)
    # res.save("res.png")

    # Return a JSON response
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=200,
    )


@app.get("/assets/{path}/{file_name}", response_class=FileResponse)
async def read_assets(path, file_name):
    return f"assets/{path}/{file_name}"

@app.get("/test", response_class=HTMLResponse)
async def read_index():
    return read_content('floorSegment.html')


@app.get("/")
def hello():
    return {"result": "Welcome to floorSegment"}



# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=8000)
