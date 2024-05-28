import numpy as np
import tritonclient.http as httpclient
from PIL import Image

# Define the server URL and model name
url = "localhost:8000"
model_name = "mnist_model"

# Create a Triton HTTP client
client = httpclient.InferenceServerClient(url=url)

# Prepare a sample image (MNIST digit)
def prepare_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np / 255.0).reshape(1, 1, 28, 28)
    return img_np

# Path to an example MNIST image
image_path = "/home/atharav/Triton/improved_integrations/test_data/7.png"

# Prepare the image
image = prepare_image(image_path)

# Create the input object
inputs = httpclient.InferInput("input.1", image.shape, "FP32")
inputs.set_data_from_numpy(image)

# Create the output object
outputs = httpclient.InferRequestedOutput("18")

# Send the request to the server
response = client.infer(model_name, inputs=[inputs], outputs=[outputs])

# Get the results
result = response.as_numpy("18")

# Print the result
print("Inference result:", result)
print("Predicted digit:", np.argmax(result))
